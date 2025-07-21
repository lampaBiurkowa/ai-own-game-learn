use std::process;

use ggez::{
    audio::{SoundSource, Source},
    event,
    glam::Vec2,
    graphics::{self, Color, DrawParam, Image},
    input::keyboard::{KeyCode, KeyInput},
    Context, GameError, GameResult,
};
use rand::Rng;

use crate::{
    client::GameClient,
    engine::{Cell, GameOverCause, TILE_SIZE},
};

pub(crate) struct Game {
    images: Images,
    client: GameClient,
    source: Source,
    move_count: u32,
    enemy_killed_count: u32,
    game_over_cause: Option<GameOverCause>,
}

impl Game {
    pub fn new(ctx: &mut Context, source: Source) -> GameResult<Self> {
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(1..=10000);
        let client = GameClient::new("http://localhost:3030", random_number);
        let images = Images {
            player: Image::from_path(ctx, "/player.png")?,
            obstacle: Image::from_path(ctx, "/obstacle.png")?,
            floor: Image::from_path(ctx, "/floor.png")?,
            enemy_vertical: Image::from_path(ctx, "/enemy-vertical.png")?,
            enemy_horizontal: Image::from_path(ctx, "/enemy-horizontal.png")?,
        };
        Ok(Self {
            images,
            client,
            source,
            move_count: 0,
            enemy_killed_count: 0,
            game_over_cause: None,
        })
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
        if let Some(cause) = &self.game_over_cause {
            let mut canvas = graphics::Canvas::from_frame(ctx, Color::from([0.0, 0.0, 0.0, 1.0]));

            let message = match cause {
                GameOverCause::Enemy => "Game Over: You were caught!",
                GameOverCause::MovementLimit => "Game Over: Movement limit reached!",
            };

            let final_text = format!(
                "{}\n\nMoves: {}\nScore: {}\nEnemies killed: {}\nPress [ESC] to exit",
                message,
                self.move_count,
                self.client.get_score(),
                self.enemy_killed_count
            );

            let text = graphics::Text::new(final_text);
            let dims = text.measure(ctx)?;
            let screen_center = Vec2::new(160.0, 160.0);
            let dest = screen_center - 0.5 * Vec2::new(dims.x, dims.y);

            canvas.draw(&text, DrawParam::default().dest(dest).color(Color::WHITE));

            canvas.finish(ctx)?;
            return Ok(());
        }

        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from([0.2, 0.2, 0.2, 1.0]));

        let viewport_grid = self.client.get_grid();
        for (y, row) in viewport_grid.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                let image = match cell {
                    Cell::Floor => &self.images.floor,
                    Cell::Obstacle => &self.images.obstacle,
                    Cell::EnemyVertical { .. } => &self.images.enemy_vertical,
                    Cell::EnemyHorizontal { .. } => &self.images.enemy_horizontal,
                    Cell::Player => &self.images.player,
                };

                let dest = Vec2::new((x as f32) * TILE_SIZE, (y as f32) * TILE_SIZE);
                canvas.draw(image, DrawParam::new().dest(dest));
            }
        }

        let display_text = graphics::Text::new(format!("Moves: {}", self.move_count));
        canvas.draw(
            &display_text,
            DrawParam::default()
                .dest(Vec2::new(10.0, 10.0))
                .color(Color::BLACK),
        );
        let score = self.client.get_score();
        let score_text = graphics::Text::new(format!("Score: {}", score));
        canvas.draw(
            &score_text,
            DrawParam::default()
                .dest(Vec2::new(10.0, 30.0))
                .color(Color::BLACK),
        );

        let kill_text = graphics::Text::new(format!("Enemies killed: {}", self.enemy_killed_count));
        canvas.draw(
            &kill_text,
            DrawParam::default()
                .dest(Vec2::new(10.0, 50.0))
                .color(Color::BLACK),
        );
        canvas.finish(ctx)?;
        Ok(())
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        input: KeyInput,
        _: bool,
    ) -> Result<(), GameError> {
        if self.game_over_cause.is_some() {
            if let Some(KeyCode::Escape) = input.keycode {
                process::exit(0);
            } else {
                return Ok(());
            }
        }

        let (dx, dy) = match input.keycode {
            Some(KeyCode::Up) => (0, -1),
            Some(KeyCode::Down) => (0, 1),
            Some(KeyCode::Left) => (-1, 0),
            Some(KeyCode::Right) => (1, 0),
            _ => (0, 0),
        };

        if dx != 0 || dy != 0 {
            self.move_count += 1;
            self.client.move_player(dx, dy);
            if self.client.enemy_killed() {
                self.enemy_killed_count += 1;
            }
            if let Some(x) = self.client.is_game_over() {
                self.game_over_cause = Some(x);
                self.source.stop(&_ctx.audio).unwrap();
            }
        }

        Ok(())
    }
}
