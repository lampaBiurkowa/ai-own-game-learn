//! Unlocking achievements through the Dibrysoft launcher.
//!
//! The launcher puts the endpoint and a per-session key in our environment when it starts the
//! game. If they are missing we were started outside the launcher, and every unlock quietly turns
//! into a no-op — achievements are a bonus, never a precondition for playing.
//!
//! Only keys declared in `.ndib/achievements.json` are accepted; anything else comes back `404`,
//! so the constants below and that file have to agree. Posting the same key twice is safe on the
//! launcher's side, and we additionally remember what this process already sent, which is what
//! lets [`Tracker::record_move`] be called after every single move without thinking about it.
//!
//! Posting happens on a background thread so a move never waits on HTTP. The flip side is that
//! anything exiting the process directly — `std::process::exit` on game over, say — has to call
//! [`flush`] first, or the unlock that just fired dies with the process.

use std::collections::HashSet;
use std::env;
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::Serialize;

use crate::engine::{Cell, GameOverCause, MAX_MOVES};

const TIMEOUT: Duration = Duration::from_secs(5);

const OFF_THE_MARK: &str = "off-the-mark";
const FIRST_CONTACT: &str = "first-contact";
const CLOSE_SHAVE: &str = "close-shave";
const GHOST_IN_THE_MACHINE: &str = "ghost-in-the-machine";
const BEELINE: &str = "beeline";
const TEN_TILES: &str = "ten-tiles";
const QUARTER_MILE: &str = "quarter-mile";
const HALFWAY_HOUSE: &str = "halfway-house";
const OUT_OF_MOVES: &str = "out-of-moves";
const TEXTBOOK_RUN: &str = "textbook-run";

/// Columns to reach for the distance achievements, and the streak "Beeline" wants. These mirror
/// the descriptions in `.ndib/achievements.json`; change one and change the other.
const TEN_TILES_AT: usize = 10;
const QUARTER_MILE_AT: usize = 25;
const HALFWAY_HOUSE_AT: usize = 50;
const BEELINE_STREAK: usize = 10;

struct Endpoint {
    url: String,
    /// Session key, valid only while this process lives.
    key: String,
}

/// Endpoint and key, read from the environment once. `None` means unlocks are unavailable.
fn endpoint() -> Option<&'static Endpoint> {
    static ENDPOINT: OnceLock<Option<Endpoint>> = OnceLock::new();

    ENDPOINT
        .get_or_init(|| {
            let url = env::var("DIBRYSOFT_ACHIEVEMENT_URL").ok()?;
            let key = env::var("DIBRYSOFT_GAME_KEY")
                .or_else(|_| env::var("DIBRYSOFT_AWARD_KEY"))
                .ok()?;
            Some(Endpoint { url, key })
        })
        .as_ref()
}

fn http() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .timeout(TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new())
    })
}

/// `true` the first time this process sees `key`, `false` on every repeat.
fn first_time(key: &str) -> bool {
    static SENT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let sent = SENT.get_or_init(|| Mutex::new(HashSet::new()));
    let mut sent = sent.lock().unwrap_or_else(|e| e.into_inner());
    sent.insert(key.to_string())
}

fn in_flight() -> &'static Mutex<Vec<JoinHandle<()>>> {
    static IN_FLIGHT: OnceLock<Mutex<Vec<JoinHandle<()>>>> = OnceLock::new();
    IN_FLIGHT.get_or_init(|| Mutex::new(Vec::new()))
}

#[derive(Serialize)]
struct UnlockRequest<'a> {
    key: &'a str,
}

/// Say once, at startup, whether unlocks are going anywhere. Optional — everything below works
/// without it — but it saves guessing when the game was started outside the launcher.
pub(crate) fn init() {
    match endpoint() {
        Some(ep) => println!("Achievements: posting to {}", ep.url),
        None => println!("Achievements: unavailable (not started by the Dibrysoft launcher)"),
    }
}

/// Wait for the posts already on their way. Call this before leaving the process, otherwise an
/// unlock that fired on the last move of a run never reaches the launcher. Bounded by the HTTP
/// timeout, so it cannot hang the shutdown for long.
// Only caller is the UI's exit path, which is commented out at the moment.
#[allow(dead_code)]
pub(crate) fn flush() {
    let handles: Vec<_> = {
        let mut in_flight = in_flight().lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *in_flight)
    };
    for handle in handles {
        let _ = handle.join();
    }
}

/// Unlock an achievement. Cheap to call from the game loop: it never blocks, never panics, and
/// posts each key at most once per run.
fn unlock(key: &'static str) {
    if endpoint().is_none() || !first_time(key) {
        return;
    }

    let spawned = std::thread::Builder::new()
        .name("dibrysoft-achievement".to_string())
        .spawn(move || post(key));

    if let Ok(handle) = spawned {
        let mut in_flight = in_flight().lock().unwrap_or_else(|e| e.into_inner());
        in_flight.retain(|h| !h.is_finished());
        in_flight.push(handle);
    }
}

fn post(key: &str) {
    let Some(ep) = endpoint() else { return };

    // No Origin header — the launcher rejects anything that looks like it came from a browser.
    let response = http()
        .post(&ep.url)
        .header("X-Dibrysoft-Award-Key", &ep.key)
        .json(&UnlockRequest { key })
        .send();

    match response {
        Ok(response) if response.status().is_success() => {}
        Ok(response) => eprintln!("achievement '{key}' refused: {}", response.status()),
        Err(e) => eprintln!("achievement '{key}' not sent: launcher unreachable ({e})"),
    }
}

/// What one move did, as far as the achievements care.
pub(crate) struct Move<'a> {
    /// Furthest column reached so far this run.
    pub(crate) score: usize,
    pub(crate) player_moved: bool,
    pub(crate) player_moved_rightwards: bool,
    pub(crate) enemy_killed: bool,
    pub(crate) game_over: Option<GameOverCause>,
    /// The board as it stands after the move, and where the player is on it.
    pub(crate) grid: &'a [Vec<Cell>],
    pub(crate) player_pos: (usize, usize),
}

/// Per-run state for the achievements that are about a sequence of moves rather than a single one.
/// Everything else is stateless — the global "already sent" set does the deduplication.
pub(crate) struct Tracker {
    rightwards_streak: usize,
}

impl Tracker {
    pub(crate) fn new() -> Self {
        Tracker {
            rightwards_streak: 0,
        }
    }

    /// Feed this after every move the player makes.
    pub(crate) fn record_move(&mut self, m: Move) {
        if m.player_moved {
            unlock(OFF_THE_MARK);
        }

        if m.player_moved_rightwards {
            self.rightwards_streak += 1;
            if self.rightwards_streak >= BEELINE_STREAK {
                unlock(BEELINE);
            }
        } else {
            self.rightwards_streak = 0;
        }

        if m.enemy_killed {
            unlock(GHOST_IN_THE_MACHINE);
        }

        // Still standing with an enemy on one of the eight surrounding tiles.
        if m.game_over.is_none() && has_adjacent_enemy(m.grid, m.player_pos) {
            unlock(CLOSE_SHAVE);
        }

        if m.score >= TEN_TILES_AT {
            unlock(TEN_TILES);
        }
        if m.score >= QUARTER_MILE_AT {
            unlock(QUARTER_MILE);
        }
        if m.score >= HALFWAY_HOUSE_AT {
            unlock(HALFWAY_HOUSE);
        }
        // Every one of the allowed moves spent going right, and none of them wasted.
        if m.score as u64 >= MAX_MOVES {
            unlock(TEXTBOOK_RUN);
        }

        match m.game_over {
            Some(GameOverCause::Enemy) => unlock(FIRST_CONTACT),
            Some(GameOverCause::MovementLimit) => unlock(OUT_OF_MOVES),
            None => {}
        }
    }
}

fn has_adjacent_enemy(grid: &[Vec<Cell>], (px, py): (usize, usize)) -> bool {
    const OFFSETS: [(isize, isize); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ];

    OFFSETS.iter().any(|&(dx, dy)| {
        let (x, y) = (px as isize + dx, py as isize + dy);
        if y < 0 || x < 0 || y as usize >= grid.len() {
            return false;
        }
        grid[y as usize]
            .get(x as usize)
            .is_some_and(|cell| matches!(cell, Cell::EnemyVertical(_) | Cell::EnemyHorizontal(_)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    /// Stands in for the launcher: takes one request, hands back what a real unlock replies with.
    fn mock_launcher() -> (u16, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(TIMEOUT)).unwrap();

            let mut request = Vec::new();
            let mut chunk = [0u8; 512];
            // Read until the JSON body has arrived; reqwest may split headers and body.
            while !request.ends_with(b"}") {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&chunk[..n]),
                }
            }

            let body = br#"{"key":"off-the-mark","name":"Off the Mark","alreadyUnlocked":false}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
            stream.flush().unwrap();

            String::from_utf8_lossy(&request).into_owned()
        });

        (port, handle)
    }

    #[test]
    fn posts_an_unlock_the_launcher_would_accept() {
        let (port, launcher) = mock_launcher();
        env::set_var(
            "DIBRYSOFT_ACHIEVEMENT_URL",
            format!("http://127.0.0.1:{port}/achievement"),
        );
        env::set_var("DIBRYSOFT_GAME_KEY", "s3cret");

        // A plain first step on an empty board: "Off the Mark" and nothing else.
        let grid = vec![vec![Cell::Floor; 4]; 4];
        Tracker::new().record_move(Move {
            score: 1,
            player_moved: true,
            player_moved_rightwards: true,
            enemy_killed: false,
            game_over: None,
            grid: &grid,
            player_pos: (1, 1),
        });
        flush();

        let request = launcher.join().unwrap();
        let lower = request.to_lowercase();

        assert!(request.starts_with("POST /achievement HTTP/1.1"), "{request}");
        assert!(lower.contains("x-dibrysoft-award-key: s3cret\r\n"), "{request}");
        assert!(lower.contains("content-type: application/json"), "{request}");
        // The launcher answers 403 to anything that looks like it came from a browser.
        assert!(!lower.contains("origin:"), "{request}");
        assert!(!lower.contains("sec-fetch-site:"), "{request}");
        assert!(request.ends_with(r#"{"key":"off-the-mark"}"#), "{request}");
    }

    #[test]
    fn an_enemy_on_a_neighbouring_tile_is_a_close_shave() {
        let mut grid = vec![vec![Cell::Floor; 4]; 4];
        grid[2][2] = Cell::EnemyHorizontal(enemy());

        assert!(has_adjacent_enemy(&grid, (1, 1)));
        assert!(has_adjacent_enemy(&grid, (3, 3)));
        assert!(!has_adjacent_enemy(&grid, (0, 0)));
    }

    #[test]
    fn a_ragged_grid_does_not_panic_at_the_edges() {
        // Rows grow as the map extends, so neighbouring rows can be different lengths.
        let mut grid = vec![vec![Cell::Floor; 2], vec![Cell::Floor; 6]];
        grid[1][5] = Cell::EnemyVertical(enemy());

        assert!(!has_adjacent_enemy(&grid, (1, 0)));
        assert!(has_adjacent_enemy(&grid, (4, 1)));
    }

    fn enemy() -> crate::engine::EnemyAttributes {
        // The fields are private and irrelevant here; go through the map generator instead.
        let grid = crate::engine::GameState::generate_map(7);
        grid.iter()
            .flatten()
            .find_map(|cell| match cell {
                Cell::EnemyVertical(attrs) | Cell::EnemyHorizontal(attrs) => Some(attrs.clone()),
                _ => None,
            })
            .expect("seed 7 spawns at least one enemy")
    }
}
