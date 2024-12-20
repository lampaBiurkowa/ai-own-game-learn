use rand::{rngs::StdRng, Rng, SeedableRng as _};
use serde::{Deserialize, Serialize};


const MAP_HEIGHT: usize = 10;
const MAP_WIDTH: usize = 10;
pub(crate) const TILE_SIZE: f32 = 32.0;
pub(crate) const VIEWPORT_WIDTH: usize = 10;
pub(crate) const VIEWPORT_HEIGHT: usize = 10;
const MAX_MOVES: usize = 100;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Cell {
    Floor,
    Obstacle,
    Player,
    EnemyVertical { pos: usize, direction: isize, range: usize },
    EnemyHorizontal { pos: usize, direction: isize, range: usize },
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct GameState {
    seed: u64,
    pub(crate) grid: Vec<Vec<Cell>>,
    pub(crate) player_pos: (usize, usize),
    score: usize,
    moves_remaining: usize,
}

impl GameState {
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
                        grid[y][x] = Cell::EnemyVertical {
                            pos: y,
                            direction: 1,
                            range: rng.gen_range(2..=6),
                        };
                    } else if enemy_type == 1 {
                        grid[y][x] = Cell::EnemyHorizontal {
                            pos: x,
                            direction: 1,
                            range: rng.gen_range(2..=6),
                        };
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
            moves_remaining: MAX_MOVES,
        }
    }
    
    pub(crate) fn move_player(&mut self, dx: isize, dy: isize) {
        if self.moves_remaining == 0 {
            println!("Game Over! Move limit reached.");
            std::process::exit(0);
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
    
                self.moves_remaining -= 1;
    
                if new_x >= VIEWPORT_WIDTH / 2 + self.grid[0].len() - VIEWPORT_WIDTH {
                    self.extend_map();
                }
    
                self.move_enemies();
            }
            Cell::EnemyVertical { .. } | Cell::EnemyHorizontal { .. } => {
                println!("Game Over! Player collided with an enemy.");
                std::process::exit(0);
            }
            _ => {}
        }
    
        if matches!(
            self.grid[py][px],
            Cell::EnemyVertical { .. } | Cell::EnemyHorizontal { .. }
        ) {
            println!("Game Over! Enemy collided with the player.");
            std::process::exit(0);
        }
    }    

    pub(crate) fn extend_map(&mut self) {
        for y in 0..MAP_HEIGHT {
            let new_cell = if rand::random::<f32>() < 0.2 {
                Cell::Obstacle
            } else if rand::random::<f32>() < 0.05 {
                Cell::EnemyVertical {
                    pos: y,
                    direction: 1,
                    range: rand::random::<usize>() % 3 + 1,
                }
            } else if rand::random::<f32>() < 0.05 {
                Cell::EnemyHorizontal {
                    pos: y,
                    direction: 1,
                    range: rand::random::<usize>() % 3 + 1,
                }
            } else {
                Cell::Floor
            };
    
            self.grid[y].push(new_cell);
        }
    }
    
    fn move_enemies(&mut self) {
        let mut updates = Vec::new();
        for y in 0..MAP_HEIGHT {
            for x in 0..self.grid[0].len() {
                match &self.grid[y][x] {
                    Cell::EnemyVertical { pos, direction, range } => {
                        let current_pos = *pos as isize;
                        let new_pos = current_pos + *direction;
                        let min_pos = (current_pos as isize - (*range as isize)).max(0);
                        let max_pos = (current_pos as isize + (*range as isize)).min((MAP_HEIGHT - 1) as isize);
    
                        if new_pos < min_pos
                            || new_pos > max_pos
                            || matches!(self.grid.get(new_pos as usize).unwrap_or(&vec![])[x], Cell::Obstacle)
                        {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                x,
                                current_pos as usize,
                                Cell::EnemyVertical {
                                    pos: current_pos as usize,
                                    direction: -*direction,
                                    range: *range,
                                },
                            ));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                x,
                                new_pos as usize,
                                Cell::EnemyVertical {
                                    pos: new_pos as usize,
                                    direction: *direction,
                                    range: *range,
                                },
                            ));
                        }
                    }
    
                    Cell::EnemyHorizontal { pos, direction, range } => {
                        let current_pos = *pos as isize;
                        let new_pos = current_pos + *direction;
    
                        let min_pos = (current_pos as isize - (*range as isize)).max(0);
                        let max_pos = (current_pos as isize + (*range as isize)).min((self.grid[0].len() - 1) as isize);
    
                        if new_pos < min_pos
                            || new_pos > max_pos
                            || matches!(self.grid[y].get(new_pos as usize).unwrap_or(&Cell::Floor), Cell::Obstacle)
                        {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                current_pos as usize,
                                y,
                                Cell::EnemyHorizontal {
                                    pos: current_pos as usize,
                                    direction: -*direction,
                                    range: *range,
                                },
                            ));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((
                                new_pos as usize,
                                y,
                                Cell::EnemyHorizontal {
                                    pos: new_pos as usize,
                                    direction: *direction,
                                    range: *range,
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    
        for (x, y, new_cell) in updates {
            self.grid[y][x] = new_cell;
        }
    }    
}
