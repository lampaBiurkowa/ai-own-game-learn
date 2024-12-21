use crate::{engine::Cell, transport::GameApiClient};

pub(crate) struct GameClient {
    api_client: GameApiClient,
    game_over: bool,
    move_successful: bool,
    enemy_killed: bool,
    moves: u64,
    grid: Vec<Vec<Cell>>,
    player_x: usize,
    player_y: usize,
}

impl GameClient {
    pub(crate) fn new(url: &str, seed: u64) -> Self {
        let api_client = GameApiClient::new(url, seed);
        let state = api_client.fetch_initial_state().unwrap();
        GameClient {
            api_client,
            game_over: false,
            move_successful: false,
            enemy_killed: false,
            moves: 0,
            grid: state.grid,
            player_x: state.player_pos.0,
            player_y: state.player_pos.1,
        }
    }

    pub(crate) fn move_player(&mut self, dx: isize, dy: isize) {
        let initial_moves = self.moves;
        let initial_grid = self.grid.clone();

        let state = self.api_client.move_player(dx, dy);
        self.moves = state.moves();
        self.move_successful = self.moves > initial_moves;
        self.game_over = state.game_over();
        self.player_x = state.player_pos.0;
        self.player_y = state.player_pos.1;
        self.grid = state.grid.clone();
        self.enemy_killed = is_enemy_killed(&initial_grid, &state.grid, (self.player_x, self.player_y));
    }

    pub(crate) fn get_grid(&self) -> Vec<Vec<Cell>> {
        self.grid.clone()
    }

    pub(crate) fn is_game_over(&self) -> bool {
        self.game_over
    }

    pub(crate) fn enemy_killed(&self) -> bool {
        self.enemy_killed
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

    if enemy_killed {
        println!("Enemy killed!");
    }
    
    enemy_killed
}
