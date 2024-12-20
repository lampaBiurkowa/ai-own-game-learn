use ggez::{event, glam::Vec2, graphics::{self, Color, DrawParam, Image}, input::keyboard::{KeyCode, KeyInput}, Context, GameError, GameResult};

use crate::{client::GameApiClient, engine::{Cell, GameState, TILE_SIZE, VIEWPORT_HEIGHT, VIEWPORT_WIDTH}};

pub(crate) struct Game {
    state: GameState,
    images: Images,
    api_client: GameApiClient,
}

impl Game {
    pub fn new(ctx: &mut Context) -> GameResult<Self> {
        let api_client = GameApiClient::new("http://localhost:3030", 1);
        let state = api_client.fetch_initial_state().unwrap();
        let images = Images {
            player: Image::from_path(ctx, "/player.png")?,
            obstacle: Image::from_path(ctx, "/obstacle.png")?,
            floor: Image::from_path(ctx, "/floor.png")?,
            enemy_vertical: Image::from_path(ctx, "/enemy-vertical.png")?,
            enemy_horizontal: Image::from_path(ctx, "/enemy-horizontal.png")?,
        };
        Ok(Self { state, images, api_client })
    }
}

struct Images {
    player: Image,
    obstacle: Image,
    floor: Image,
    enemy_vertical: Image,
    enemy_horizontal: Image,
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

                if world_x >= self.state.grid[0].len() {
                    continue;
                }

                let image = match self.state.grid[y][world_x] {
                    Cell::Floor => &self.images.floor,
                    Cell::Obstacle => &self.images.obstacle,
                    Cell::Player => &self.images.player,
                    Cell::EnemyVertical { .. } => &self.images.enemy_vertical,
                    Cell::EnemyHorizontal { .. } => &self.images.enemy_horizontal,
                };

                let dest = Vec2::new((x as f32) * TILE_SIZE, (y as f32) * TILE_SIZE);
                canvas.draw(image, DrawParam::new().dest(dest));
            }
        }

        canvas.finish(ctx)?;
        Ok(())
    }

    fn key_down_event(&mut self, _ctx: &mut Context, input: KeyInput, _: bool) -> Result<(), GameError> {
        let (dx, dy) = match input.keycode {
            Some(KeyCode::Up) => (0, -1),
            Some(KeyCode::Down) => (0, 1),
            Some(KeyCode::Left) => (-1, 0),
            Some(KeyCode::Right) => (1, 0),
            _ => (0, 0),
        };

        if dx != 0 || dy != 0 {
            self.state = self.api_client.move_player(dx, dy);
            self.state = self.api_client.fetch_state();
        }

        Ok(())
    }
}
