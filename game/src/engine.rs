use rand::{rngs::StdRng, Rng, SeedableRng as _};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAP_HEIGHT: usize = 10;
const MAP_WIDTH: usize = 10;
pub(crate) const TILE_SIZE: f32 = 32.0;
pub(crate) const VIEWPORT_WIDTH: usize = 10;
pub(crate) const VIEWPORT_HEIGHT: usize = 10;
const MAX_MOVES: u64 = 100;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct EnemyAttributes {
    pos: usize,
    direction: isize,
    range: usize,
    initial_pos: usize,
    id: String
}

impl EnemyAttributes {
    pub(crate) fn id(&self) -> String {
        self.id.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Cell {
    Floor,
    Obstacle,
    Player,
    EnemyVertical(EnemyAttributes),
    EnemyHorizontal(EnemyAttributes),
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct GameState {
    seed: u64,
    pub(crate) grid: Vec<Vec<Cell>>,
    pub(crate) player_pos: (usize, usize),
    score: usize,
    moves: u64,
    game_over: bool
}

fn is_valid_vertical_spawn(grid: &[Vec<Cell>], x: usize, y: usize, range: usize) -> bool {
    if x >= grid[0].len() {
        return false;
    }

    let min_y = y.saturating_sub(range);
    let max_y = (y + range).min(MAP_HEIGHT - 1);

    for yy in min_y..=max_y {
        if matches!(grid[yy][x], Cell::EnemyVertical(_) | Cell::EnemyHorizontal(_) | Cell::Obstacle) {
            return false;
        }
    }

    true
}

fn is_valid_horizontal_spawn(grid: &[Vec<Cell>], x: usize, y: usize, range: usize) -> bool {
    if y >= MAP_HEIGHT {
        return false;
    }

    let min_x = x.saturating_sub(range);
    let max_x = (x + range).min(grid[y].len() - 1);

    for xx in min_x..=max_x {
        if matches!(grid[y][xx], Cell::EnemyVertical(_) | Cell::EnemyHorizontal(_)) {
            return false;
        }
    }
    true
}

impl GameState {
    pub(crate) fn game_over(&self) -> bool {
        self.game_over
    }

    pub(crate) fn moves(&self) -> u64 {
        self.moves
    }

    pub(crate) fn score(&self) -> usize {
        self.score
    }

    fn generate_map(seed: u64) -> Vec<Vec<Cell>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut grid = vec![vec![Cell::Floor; MAP_WIDTH]; MAP_HEIGHT];
    
        for y in 0..MAP_HEIGHT {
            for x in 0..MAP_WIDTH {
                if rng.gen_bool(0.2) {
                    grid[y][x] = Cell::Obstacle;
                } else {
                    let enemy_type = rng.gen_range(0..10);
    
                    if enemy_type == 0 {
                        let range = rng.gen_range(2..=6);
                        if !is_valid_vertical_spawn(&grid, x, y, range) {
                            continue;
                        }
                        grid[y][x] = Cell::EnemyVertical(EnemyAttributes {
                            pos: y,
                            direction: 1,
                            range,
                            initial_pos: y,
                            id: Uuid::new_v4().to_string()
                        });
                    } else if enemy_type == 1 {
                        let range = rng.gen_range(2..=6);
                        if !is_valid_horizontal_spawn(&grid, x, y, range) {
                            continue;
                        }
                        grid[y][x] = Cell::EnemyHorizontal(EnemyAttributes {
                            pos: x,
                            direction: 1,
                            range,
                            initial_pos: x,
                            id: Uuid::new_v4().to_string()
                        });
                    }
                }
            }
        }
    
        grid[MAP_HEIGHT / 2][0] = Cell::Player;
        grid
    }
    

    pub(crate) fn new_with_seed(seed: u64) -> Self {
        let grid = Self::generate_map(seed);
        GameState {
            seed,
            grid,
            player_pos: (0, MAP_HEIGHT / 2),
            score: 0,
            moves: 0,
            game_over: false
        }
    }

    pub(crate) fn move_player(&mut self, dx: isize, dy: isize) {
        if self.moves >= MAX_MOVES {
            self.game_over = true;
            println!("Game Over! Move limit reached.");
            return;
        }

        let (px, py) = self.player_pos;
        let new_x = (px as isize + dx).clamp(0, (self.grid[0].len() - 1) as isize) as usize;
        let new_y = (py as isize + dy).clamp(0, (MAP_HEIGHT - 1) as isize) as usize;

        match self.grid[new_y][new_x] {
            Cell::Floor => {
                self.grid[py][px] = Cell::Floor;
                self.grid[new_y][new_x] = Cell::Player;
                self.player_pos = (new_x, new_y);

                if new_x > self.score {
                    self.score = new_x;
                }

                self.moves += 1;

                if new_x >= VIEWPORT_WIDTH / 2 + self.grid[0].len() - VIEWPORT_WIDTH {
                    self.extend_map();
                }

                self.move_enemies();
            }
            Cell::EnemyVertical(_) | Cell::EnemyHorizontal(_) => {
                self.game_over = true;
                println!("Game Over! Player collided with an enemy.");
                return;
            }
            _ => {}
        }
    }

    pub(crate) fn extend_map(&mut self) {
        let mut rng = StdRng::seed_from_u64(self.seed + self.moves);
        for y in 0..MAP_HEIGHT {
            self.grid[y].push(Cell::Floor);
        }
    
        for y in 0..MAP_HEIGHT {
            let new_x = self.grid[y].len() - 1;
            let random_value = rng.gen::<f32>();
    
            if random_value < 0.3 {
                self.grid[y][new_x] = Cell::Obstacle;
            } else if random_value < 0.5 {
                let range = rng.gen_range(2..=6);
                if is_valid_vertical_spawn(&self.grid, new_x, y, range) {
                    self.grid[y][new_x] = Cell::EnemyVertical(EnemyAttributes {
                        pos: y,
                        direction: 1,
                        range,
                        initial_pos: y,
                        id: Uuid::new_v4().to_string()
                    });
                }
            } else if random_value < 0.7 {
                let range = rng.gen_range(2..=6);
                if is_valid_horizontal_spawn(&self.grid, new_x, y, range) {
                    self.grid[y][new_x] = Cell::EnemyHorizontal(EnemyAttributes {
                        pos: new_x,
                        direction: 1,
                        range,
                        initial_pos: new_x,
                        id: Uuid::new_v4().to_string()
                    });
                }
            }
        }
    }
    
    
    fn move_enemies(&mut self) {
        let mut updates = Vec::new();
    
        for y in 0..MAP_HEIGHT {
            let row_len = self.grid[y].len();
            for x in 0..row_len {
                match &self.grid[y][x] {
                    Cell::EnemyVertical(attrs) => {
                        let new_pos = (attrs.pos as isize + attrs.direction) as usize;
                        let min_pos = attrs.initial_pos.saturating_sub(attrs.range);
                        let max_pos = attrs.initial_pos + attrs.range;
    
                        if new_pos < min_pos
                            || new_pos > max_pos
                            || new_pos >= MAP_HEIGHT
                            || matches!(self.grid[new_pos][x], Cell::Obstacle)
                        {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                x,
                                attrs.pos,
                                Cell::EnemyVertical(EnemyAttributes {
                                    pos: attrs.pos,
                                    direction: -attrs.direction,
                                    range: attrs.range,
                                    initial_pos: attrs.initial_pos,
                                    id: attrs.id()
                                }),
                            ));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                x,
                                new_pos,
                                Cell::EnemyVertical(EnemyAttributes {
                                    pos: new_pos,
                                    direction: attrs.direction,
                                    range: attrs.range,
                                    initial_pos: attrs.initial_pos,
                                    id: attrs.id()
                                }),
                            ));
                        }
                    }
                    Cell::EnemyHorizontal(attrs) => {
                        let new_pos = (attrs.pos as isize + attrs.direction) as usize;
                        let min_pos = attrs.initial_pos.saturating_sub(attrs.range);
                        let max_pos = attrs.initial_pos + attrs.range;
    
                        if new_pos < min_pos
                            || new_pos > max_pos
                            || new_pos >= self.grid[y].len()
                            || matches!(self.grid[y][new_pos], Cell::Obstacle)
                        {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                attrs.pos,
                                y,
                                Cell::EnemyHorizontal(EnemyAttributes {
                                    pos: attrs.pos,
                                    direction: -attrs.direction,
                                    range: attrs.range,
                                    initial_pos: attrs.initial_pos,
                                    id: attrs.id()
                                }),
                            ));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                new_pos,
                                y,
                                Cell::EnemyHorizontal(EnemyAttributes {
                                    pos: new_pos,
                                    direction: attrs.direction,
                                    range: attrs.range,
                                    initial_pos: attrs.initial_pos,
                                    id: attrs.id()
                                }),
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    
        for (x, y, new_cell) in updates {
            if self.grid[y][x] != Cell::Player {
                self.grid[y][x] = new_cell;
            }
        }
    }    
}
