use ggez::{
    event,
    glam::Vec2,
    graphics::{self, Color, DrawParam, Image},
    input::keyboard::{KeyCode, KeyInput},
    Context, GameError, GameResult,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, path};

const MAP_HEIGHT: usize = 10;
const TILE_SIZE: f32 = 32.0;
const VIEWPORT_WIDTH: usize = 10;
const VIEWPORT_HEIGHT: usize = 10;

#[derive(Clone, Debug, PartialEq)]
enum Cell {
    Floor,
    Obstacle,
    Player,
    EnemyVertical { pos: usize, direction: isize, range: usize },
    EnemyHorizontal { pos: usize, direction: isize, range: usize },
}

struct GameMap {
    seed: u64,
    rng: StdRng,
    grid: Vec<Vec<Cell>>,
}

impl GameMap {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut grid = vec![vec![Cell::Floor; VIEWPORT_WIDTH]; MAP_HEIGHT];

        for y in 0..MAP_HEIGHT {
            for x in 0..VIEWPORT_WIDTH {
                if rng.gen_bool(0.2) {
                    grid[y][x] = Cell::Obstacle;
                } else if rng.gen_bool(0.05) {
                    grid[y][x] = Cell::EnemyVertical {
                        pos: y,
                        direction: 1,
                        range: rng.gen_range(1..=3),
                    };
                } else if rng.gen_bool(0.05) {
                    grid[y][x] = Cell::EnemyHorizontal {
                        pos: x,
                        direction: 1,
                        range: rng.gen_range(1..=3),
                    };
                }
            }
        }

        grid[MAP_HEIGHT / 2][0] = Cell::Player;
        Self { seed, rng, grid }
    }

    fn extend_map(&mut self) {
        let new_columns = VIEWPORT_WIDTH; // Number of new columns to add
        let old_width = self.grid[0].len();
        let new_width = old_width + new_columns;

        for y in 0..MAP_HEIGHT {
            for x in old_width..new_width {
                let new_cell = if self.rng.gen_bool(0.2) {
                    Cell::Obstacle
                } else if self.rng.gen_bool(0.05) {
                    Cell::EnemyVertical {
                        pos: y,
                        direction: 1,
                        range: self.rng.gen_range(1..=3),
                    }
                } else if self.rng.gen_bool(0.05) {
                    Cell::EnemyHorizontal {
                        pos: x,
                        direction: 1,
                        range: self.rng.gen_range(1..=3),
                    }
                } else {
                    Cell::Floor
                };

                self.grid[y].push(new_cell);
            }
        }
    }

    fn is_valid_position(&self, x: usize, y: usize) -> bool {
        x < self.grid[0].len() && y < MAP_HEIGHT && matches!(self.grid[y][x], Cell::Floor)
    }

    fn move_player(&mut self, old_x: usize, old_y: usize, new_x: usize, new_y: usize) -> bool {
        if self.is_valid_position(new_x, new_y) {
            self.grid[old_y][old_x] = Cell::Floor;
            self.grid[new_y][new_x] = Cell::Player;
            true
        } else {
            false
        }
    }

    fn move_enemies(&mut self) {
        let mut updates = Vec::new();

        for y in 0..MAP_HEIGHT {
            for x in 0..self.grid[0].len() {
                match &self.grid[y][x] {
                    Cell::EnemyVertical { pos, direction, range } => {
                        let new_pos = (*pos as isize + *direction) as usize;
                        let range_start = pos.saturating_sub(*range);
                        let range_end = pos.saturating_add(*range);

                        if new_pos < range_start || new_pos >= MAP_HEIGHT || new_pos > range_end || !self.is_valid_position(x, new_pos) {
                            updates.push((x, y, Cell::EnemyVertical { pos: *pos, direction: -*direction, range: *range }));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((x, new_pos, Cell::EnemyVertical { pos: new_pos, direction: *direction, range: *range }));
                        }
                    }
                    Cell::EnemyHorizontal { pos, direction, range } => {
                        let new_pos = (*pos as isize + *direction) as usize;
                        let range_start = pos.saturating_sub(*range);
                        let range_end = pos.saturating_add(*range);

                        if new_pos < range_start || new_pos >= self.grid[0].len() || new_pos > range_end || !self.is_valid_position(new_pos, y) {
                            updates.push((x, y, Cell::EnemyHorizontal { pos: *pos, direction: -*direction, range: *range }));
                        } else {
                            updates.push((x, y, Cell::Floor));
                            updates.push((new_pos, y, Cell::EnemyHorizontal { pos: new_pos, direction: *direction, range: *range }));
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

struct GameState {
    map: GameMap,
    player_pos: (usize, usize),
    images: Images,
}

struct Images {
    player: Image,
    obstacle: Image,
    floor: Image,
    enemy_vertical: Image,
    enemy_horizontal: Image,
}

impl GameState {
    fn new(ctx: &mut Context, seed: u64) -> GameResult<Self> {
        let map = GameMap::new(seed);
        let player_pos = (0, MAP_HEIGHT / 2);

        let images = Images {
            player: Image::from_path(ctx, "/player.png")?,
            obstacle: Image::from_path(ctx, "/obstacle.png")?,
            floor: Image::from_path(ctx, "/floor.png")?,
            enemy_vertical: Image::from_path(ctx, "/enemy-vertical.png")?,
            enemy_horizontal: Image::from_path(ctx, "/enemy-horizontal.png")?,
        };

        Ok(Self { map, player_pos, images })
    }

    fn move_player(&mut self, dx: isize, dy: isize) {
        let (px, py) = self.player_pos;
        let new_x = (px as isize + dx).clamp(0, (self.map.grid[0].len() - 1) as isize) as usize;
        let new_y = (py as isize + dy).clamp(0, (MAP_HEIGHT - 1) as isize) as usize;

        match self.map.grid[new_y][new_x] {
            Cell::EnemyVertical { .. } | Cell::EnemyHorizontal { .. } => {
                std::process::exit(0);
            }
            _ => {}
        }

        if self.map.move_player(px, py, new_x, new_y) {
            self.player_pos = (new_x, new_y);
            if new_x >= self.map.grid[0].len() - VIEWPORT_WIDTH {
                self.map.extend_map();
            }
            self.map.move_enemies();
        }
    }
}

struct Game {
    state: GameState,
}

impl Game {
    pub fn new(ctx: &mut Context) -> GameResult<Self> {
        let state = GameState::new(ctx, 1)?;
        Ok(Self { state })
    }
}

impl event::EventHandler<ggez::GameError> for Game {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from([0.2, 0.2, 0.2, 1.0]));

        for y in 0..VIEWPORT_HEIGHT {
            for x in 0..VIEWPORT_WIDTH {
                let viewport_offset = if self.state.player_pos.0 < VIEWPORT_WIDTH / 2 {
                    0
                } else {
                    self.state.player_pos.0 - VIEWPORT_WIDTH / 2
                };
                let world_x = viewport_offset + x;

                if world_x >= self.state.map.grid[0].len() {
                    continue;
                }

                let image = match self.state.map.grid[y][world_x] {
                    Cell::Floor => &self.state.images.floor,
                    Cell::Obstacle => &self.state.images.obstacle,
                    Cell::Player => &self.state.images.player,
                    Cell::EnemyVertical { .. } => &self.state.images.enemy_vertical,
                    Cell::EnemyHorizontal { .. } => &self.state.images.enemy_horizontal,
                };

                let dest = Vec2::new((x as f32) * TILE_SIZE, (y as f32) * TILE_SIZE);
                canvas.draw(image, DrawParam::new().dest(dest));
            }
        }

        canvas.finish(ctx)?;
        Ok(())
    }

    fn key_down_event(&mut self, _ctx: &mut Context, input: KeyInput, _: bool) -> Result<(), GameError> {
        match input.keycode {
            Some(KeyCode::Up) => self.state.move_player(0, -1),
            Some(KeyCode::Down) => self.state.move_player(0, 1),
            Some(KeyCode::Left) => self.state.move_player(-1, 0),
            Some(KeyCode::Right) => self.state.move_player(1, 0),
            _ => (),
        };

        Ok(())
    }
}

pub fn main() -> GameResult {
    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let cb = ggez::ContextBuilder::new("simple_game", "ggez").add_resource_path(resource_dir);
    let (mut ctx, events_loop) = cb.build()?;
    let game = Game::new(&mut ctx)?;
    event::run(ctx, events_loop, game)
}