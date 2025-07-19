use ggez::{
    glam::Vec2,
    graphics::{self, Canvas, Color, DrawParam, Drawable, Text},
    input::keyboard::{KeyCode, KeyInput},
    Context, GameError, GameResult,
};

use crate::{ui::Game, AppState};

use rand::Rng;

struct SquareParticle {
    pos: Vec2,
    vel: Vec2,
    size: f32,
    color: Color,
}

pub struct MenuState {
    selected: usize,
    options: Vec<String>,
    particles: Vec<SquareParticle>,
}

impl MenuState {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut particles = Vec::new();

        for _ in 0..50 {
            particles.push(SquareParticle {
                pos: Vec2::new(rng.gen_range(0.0..320.0), rng.gen_range(0.0..320.0)),
                vel: Vec2::new(rng.gen_range(-0.7..0.7), rng.gen_range(-0.7..0.7)),
                size: rng.gen_range(8.0..20.0),
                color: Color::from_rgb(
                    rng.gen_range(50..200),
                    rng.gen_range(50..200),
                    rng.gen_range(50..200),
                ),
            });
        }

        Self {
            options: vec!["Start".to_string(), "Quit".to_string()],
            selected: 0,
            particles,
        }
    }

    pub fn handle_input(
        &mut self,
        ctx: &mut Context,
        input: KeyInput,
    ) -> Result<Option<AppState>, GameError> {
        match input.keycode {
            Some(KeyCode::Up) => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }
            Some(KeyCode::Down) => {
                if self.selected < self.options.len() - 1 {
                    self.selected += 1;
                }
            }
            Some(KeyCode::Return) => match self.selected {
                0 => {
                    // ✅ Now works because ctx is passed in
                    let game = Game::new(ctx)?;
                    return Ok(Some(AppState::Game(game)));
                }
                1 => std::process::exit(0),
                _ => {}
            },
            _ => {}
        }

        Ok(None)
    }

    pub fn update(&mut self, _ctx: &mut Context) -> GameResult {
        for p in &mut self.particles {
            p.pos += p.vel;
            if p.pos.x < 0.0 || p.pos.x > 320.0 - p.size {
                p.vel.x *= -1.0;
            }
            if p.pos.y < 0.0 || p.pos.y > 320.0 - p.size {
                p.vel.y *= -1.0;
            }
        }
        Ok(())
    }

    pub fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::BLACK);
        let screen_w = 320.0;

        for x in (0..=320).step_by(32) {
            let line = graphics::Mesh::new_line(
                ctx,
                &[Vec2::new(x as f32, 0.0), Vec2::new(x as f32, 320.0)],
                1.0,
                Color::from_rgb(40, 40, 40),
            )?;
            canvas.draw(&line, DrawParam::default());
        }

        for y in (0..=320).step_by(32) {
            let line = graphics::Mesh::new_line(
                ctx,
                &[Vec2::new(0.0, y as f32), Vec2::new(320.0, y as f32)],
                1.0,
                Color::from_rgb(40, 40, 40),
            )?;
            canvas.draw(&line, DrawParam::default());
        }

        for p in &self.particles {
            let rect = graphics::Rect::new(p.pos.x, p.pos.y, p.size, p.size);
            let mesh = graphics::Mesh::new_rectangle(
                ctx,
                graphics::DrawMode::fill(),
                rect,
                p.color,
            )?;
            canvas.draw(&mesh, DrawParam::default());
        }


        let title = graphics::Text::new(("Chasin' Blocks"));
        let dest = Vec2::new(40.0, 40.0);
        canvas.draw(&title, DrawParam::default().dest(dest));

        for (i, option) in self.options.iter().enumerate() {
            let mut fragment = graphics::TextFragment::new(option.clone());

            if i == self.selected {
                fragment = fragment.color(Color::YELLOW);
            } else {
                fragment = fragment.color(Color::WHITE);
            }

            let text = Text::new(fragment);

            let x = screen_w / 2.0 - text.dimensions(ctx).unwrap().x as f32 / 2.0;
            let y = 100.0 + i as f32 * 40.0;

            canvas.draw(&text, DrawParam::default().dest(Vec2::new(x, y)));
        }

        canvas.finish(ctx)?;
        Ok(())
    }
}
