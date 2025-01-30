use crate::{engine::{Cell, GameOverCause}, transport::GameApiClient};

pub(crate) struct GameClient {
    api_client: GameApiClient,
    game_over: Option<GameOverCause>,
    move_successful: bool,
    enemy_killed: bool,
    moves: u64,
    grid: Vec<Vec<Cell>>,
    player_x: usize,
    player_y: usize,
    score: usize,
    score_increased: bool,
    player_moved: bool,
    player_moved_rightwards: bool
}

const VIEWPORT_WIDTH: usize = 10;
const VIEWPORT_HEIGHT: usize = 10;

impl GameClient {
    pub(crate) fn new(url: &str, seed: u64) -> Self {
        let api_client = GameApiClient::new(url, seed);
        let state = api_client.fetch_initial_state().unwrap();
        GameClient {
            api_client,
            game_over: None,
            move_successful: false,
            enemy_killed: false,
            moves: 0,
            grid: state.grid,
            player_x: state.player_pos.0,
            player_y: state.player_pos.1,
            score: 0,
            score_increased: false,
            player_moved: false,
            player_moved_rightwards: false,
        }
    }

    pub(crate) fn move_player(&mut self, dx: isize, dy: isize) {
        let initial_moves = self.moves;
        let initial_score = self.score;
        let initial_grid = self.grid.clone();

        let state = self.api_client.move_player(dx, dy);
        self.moves = state.moves();
        self.move_successful = self.moves > initial_moves;
        self.game_over = state.game_over();
        self.player_moved = self.player_x != state.player_pos.0 || self.player_y != state.player_pos.1;
        self.player_moved_rightwards = state.player_pos.0 > self.player_x;
        self.player_x = state.player_pos.0;
        self.player_y = state.player_pos.1;
        self.grid = state.grid.clone();
        self.score = state.score();
        self.score_increased = self.score > initial_score;
        self.enemy_killed = is_enemy_killed(&initial_grid, &state.grid, (self.player_x, self.player_y));
    }

    pub(crate) fn get_grid(&self) -> Vec<Vec<Cell>> {
        let grid_height = self.grid.len();
        let grid_width = if grid_height > 0 { self.grid[0].len() } else { 0 };
    
        let mut start_x = if self.player_x < VIEWPORT_WIDTH / 2 {
            0
        } else {
            self.player_x - VIEWPORT_WIDTH / 2
        };
        let mut end_x = start_x + VIEWPORT_WIDTH;
    
        let mut start_y = if self.player_y < VIEWPORT_HEIGHT / 2 {
            0
        } else {
            self.player_y - VIEWPORT_HEIGHT / 2
        };
        let mut end_y = start_y + VIEWPORT_HEIGHT;
    
        if end_x > grid_width {
            end_x = grid_width;
            if end_x >= VIEWPORT_WIDTH {
                start_x = end_x - VIEWPORT_WIDTH;
            }
        }
    
        if end_y > grid_height {
            end_y = grid_height;
            if end_y >= VIEWPORT_HEIGHT {
                start_y = end_y - VIEWPORT_HEIGHT;
            }
        }
    
        self.grid[start_y..end_y]
            .iter()
            .map(|row| row[start_x..end_x].to_vec())
            .collect()
    }
       

    pub(crate) fn get_score(&self) -> usize {
        self.score
    }

    pub(crate) fn score_increased(&self) -> bool {
        self.score_increased
    }

    pub(crate) fn is_game_over(&self) -> Option<GameOverCause> {
        self.game_over.clone()
    }

    pub(crate) fn enemy_killed(&self) -> bool {
        self.enemy_killed
    }

    pub(crate) fn player_moved(&self) -> bool {
        self.player_moved
    }

    pub(crate) fn player_moved_rightwards(&self) -> bool {
        self.player_moved_rightwards
    }

    pub(crate) fn get_player_position(&self) -> (usize, usize) {
        (self.player_x, self.player_y)
    }
}

fn is_enemy_killed(
    grid_before: &[Vec<Cell>],
    grid_after: &[Vec<Cell>],
    player_pos: (usize, usize),
) -> bool {
    let (px, py) = player_pos;
    let offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1), (0, 1), (1, 1),
    ];

    let within_bounds = |x: isize, y: isize, grid: &[Vec<Cell>]| -> bool {
        y >= 0 && y < grid.len() as isize && x >= 0 && x < grid[0].len() as isize
    };
    let enemies_before: Vec<String> = offsets
        .iter()
        .filter_map(|&(dx, dy)| {
            let nx = px as isize + dx;
            let ny = py as isize + dy;
            if within_bounds(nx, ny, grid_before) {
                match &grid_before[ny as usize][nx as usize] {
                    Cell::EnemyVertical(attrs) | Cell::EnemyHorizontal(attrs) => Some(attrs.id()),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect();

    let enemies_after: Vec<String> = grid_after
        .iter()
        .flat_map(|row| row.iter())
        .filter_map(|cell| match cell {
            Cell::EnemyVertical(attrs) | Cell::EnemyHorizontal(attrs) => Some(attrs.id()),
            _ => None,
        })
        .collect();

    let enemy_killed = enemies_before
        .into_iter()
        .any(|id| !enemies_after.contains(&id));

    enemy_killed
}
