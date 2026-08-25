use ggez::{
    audio::{SoundSource, Source},
    glam::Vec2,
    graphics::{self, Canvas, Color, DrawParam, Image, Text, TextFragment},
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

enum MenuPhase {
    SplashStudio,
    SplashGameTitle,
    MainMenu,
}

pub struct MenuState {
    selected: usize,
    options: Vec<String>,
    particles: Vec<SquareParticle>,
    source: Source,
    showing_credits: bool,
    credits_offset: f32,
    phase: MenuPhase,
    splash_timer: f32,
    box_open: f32,
    intro_particles: Vec<SquareParticle>,
    fade_alpha: f32,
    block_chaser_pos: Vec2,
    block_target_path: Vec<Vec2>,
    block_path_index: usize,
    block_timer: f32,
    destruction_particles: Vec<SquareParticle>,
    fade_image: Image,
}

impl MenuState {
    pub fn new(source: Source, fade_image: Image) -> Self {
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
        let mut intro_particles = Vec::new();
        for _ in 0..70 {
            intro_particles.push(SquareParticle {
                pos: Vec2::new(160.0, 160.0),
                vel: Vec2::new(rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0)),
                size: rng.gen_range(2.0..5.0),
                color: Color::from_rgb(
                    rng.gen_range(40..255),
                    rng.gen_range(40..255),
                    rng.gen_range(40..255),
                ),
            });
        }

        Self {
            options: vec![
                "Start".to_string(),
                "Credits".to_string(),
                "Quit".to_string(),
            ],
            selected: 0,
            particles,
            source,
            showing_credits: false,
            credits_offset: 320.0,
            phase: MenuPhase::SplashStudio,
            splash_timer: 0.0,
            box_open: 0.0,
            intro_particles,
            fade_alpha: 1.0,
            block_chaser_pos: Vec2::new(40.0, 80.0),
            block_target_path: vec![
                Vec2::new(40.0, 80.0),
                Vec2::new(260.0, 80.0),
                Vec2::new(260.0, 200.0),
                Vec2::new(40.0, 200.0),
            ],
            block_path_index: 1,
            block_timer: 0.0,
            destruction_particles: Vec::new(),
            fade_image,
        }
    }

    pub fn handle_input(
        &mut self,
        ctx: &mut Context,
        input: KeyInput,
    ) -> Result<Option<AppState>, GameError> {
        if !matches!(self.phase, MenuPhase::MainMenu) {
            return Ok(None); // Ignore input during splash
        }
        match input.keycode {
            Some(KeyCode::Escape) => {
                if self.showing_credits {
                    self.showing_credits = false;
                }
            }
            Some(KeyCode::Up) if !self.showing_credits => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }
            Some(KeyCode::Down) if !self.showing_credits => {
                if self.selected < self.options.len() - 1 {
                    self.selected += 1;
                }
            }
            Some(KeyCode::Return) if !self.showing_credits => match self.selected {
                0 => {
                    self.source.stop(&ctx.audio).unwrap();
                    let mut music = Source::new(&ctx.audio, "/game.mp3")?;
                    music.set_repeat(true);
                    music.play(&ctx.audio).unwrap();
                    let game = Game::new(ctx, music)?;
                    return Ok(Some(AppState::Game(game)));
                }
                1 => {
                    self.showing_credits = true;
                }
                2 => std::process::exit(0),
                _ => {}
            },
            _ => {}
        }

        Ok(None)
    }

    pub fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = ctx.time.delta().as_secs_f32();
        match self.phase {
            MenuPhase::SplashStudio => {
                self.splash_timer += dt;
                self.box_open = (self.splash_timer / 3.0).min(1.0);
                if self.splash_timer < 2.0 {
                    self.fade_alpha = 1.0 - (self.splash_timer / 2.0);
                } else if self.splash_timer < 6.0 {
                    self.fade_alpha = 0.0;
                } else if self.splash_timer < 7.0 {
                    self.fade_alpha = ((self.splash_timer - 6.0) / 1.0).min(1.0);
                }

                if self.splash_timer > 3.0 {
                    for p in &mut self.intro_particles {
                        p.pos += p.vel;
                        p.vel *= 0.98;
                    }
                }

                if self.splash_timer > 7.0 {
                    self.phase = MenuPhase::SplashGameTitle;
                    self.splash_timer = 0.0;
                }

                return Ok(());
            }
            MenuPhase::SplashGameTitle => {
                let dt = ctx.time.delta().as_secs_f32();
                self.block_timer += dt;

                let speed = 100.0;
                let target = self.block_target_path[self.block_path_index];
                let dir = (target - self.block_chaser_pos).normalize_or_zero();
                self.block_chaser_pos += dir * speed * dt;
                if self.block_chaser_pos.distance(target) < 4.0 {
                    for _ in 0..30 {
                        self.destruction_particles.push(SquareParticle {
                            pos: self.block_target_path[self.block_path_index],
                            vel: Vec2::new(
                                rand::random::<f32>() * 2.0 - 1.0,
                                rand::random::<f32>() * 2.0 + 1.0,
                            ),
                            size: rand::random::<f32>() * 3.0 + 2.0,
                            color: Color::from_rgb(50, 120, 255),
                        });
                    }
                    if self.block_path_index < self.block_target_path.len() - 1 {
                        self.block_path_index += 1;
                    } else {
                        self.block_chaser_pos = self.block_target_path[self.block_path_index];
                    }
                }

                for p in &mut self.destruction_particles {
                    p.pos += p.vel;
                    p.vel.y += 0.1; // gravity
                }

                if self.block_timer >= 8.0 {
                    self.phase = MenuPhase::MainMenu;
                    self.splash_timer = 0.0;
                }

                Ok(())
            }
            MenuPhase::MainMenu => {
                for p in &mut self.particles {
                    p.pos += p.vel;
                    if p.pos.x < 0.0 || p.pos.x > 320.0 - p.size {
                        p.vel.x *= -1.0;
                    }
                    if p.pos.y < 0.0 || p.pos.y > 320.0 - p.size {
                        p.vel.y *= -1.0;
                    }
                }

                if self.showing_credits {
                    self.credits_offset -= 0.5;
                    if self.credits_offset < -320.0 {
                        self.credits_offset = 320.0;
                    }
                }

                Ok(())
            }
        }
    }

    pub fn draw(&mut self, ctx: &mut Context) -> GameResult {
        match self.phase {
            MenuPhase::SplashStudio => {
                let mut canvas = Canvas::from_frame(ctx, Color::BLACK);

                let box_color = Color::from_rgb(160, 110, 50);
                let body = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    graphics::Rect::new(110.0, 110.0, 100.0, 100.0),
                    box_color,
                )?;
                canvas.draw(&body, DrawParam::default());

                let flap = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    graphics::Rect::new(0.0, 0.0, 100.0, 20.0),
                    box_color,
                )?;

                let flap_angle = self.box_open * std::f32::consts::FRAC_PI_4;

                // Left flap
                canvas.draw(
                    &flap,
                    DrawParam::default()
                        .dest(Vec2::new(110.0, 110.0))
                        .rotation(-flap_angle)
                        .offset(Vec2::new(0.0, 1.0)),
                );

                // Right flap
                canvas.draw(
                    &flap,
                    DrawParam::default()
                        .dest(Vec2::new(210.0, 110.0))
                        .rotation(flap_angle)
                        .offset(Vec2::new(1.0, 1.0)),
                );

                if self.splash_timer > 3.0 {
                    for p in &self.intro_particles {
                        let rect = graphics::Rect::new(p.pos.x, p.pos.y, p.size, p.size);
                        let mesh = graphics::Mesh::new_rectangle(
                            ctx,
                            graphics::DrawMode::fill(),
                            rect,
                            p.color,
                        )?;
                        canvas.draw(&mesh, DrawParam::default());
                    }
                }

                if self.splash_timer > 2.0 {
                    let t = ((self.splash_timer - 2.0) * 1.2).min(1.0);
                    let bounce = (t * std::f32::consts::PI).sin() * 10.0;
                    let text = Text::new(
                        TextFragment::new("Cardboard Studio")
                            .scale(32.0)
                            .color(Color::WHITE),
                    );
                    let dims = text.measure(ctx)?;
                    let x = 160.0 - dims.x / 2.0;
                    let y = 240.0 - bounce;
                    canvas.draw(&text, DrawParam::default().dest(Vec2::new(x, y)));
                }

                if self.splash_timer > 3.5 {
                    let text = Text::new(
                        TextFragment::new("presents")
                            .scale(20.0)
                            .color(Color::WHITE),
                    );
                    let dims = text.measure(ctx)?;
                    let x = 160.0 - dims.x / 2.0;
                    let y = 270.0;

                    // Draw the text fully
                    canvas.draw(&text, DrawParam::default().dest(Vec2::new(x, y)));

                    // Overlay fade-out rectangle
                    let time_since = self.splash_timer - 3.5;
                    let alpha = (1.0 - (time_since / 1.0)).clamp(0.0, 1.0);

                    let fade_rect = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        graphics::Rect::new(x, y, dims.x, dims.y),
                        Color::new(0.0, 0.0, 0.0, alpha),
                    )?;
                    canvas.draw(&fade_rect, DrawParam::default());
                }

                if self.fade_alpha > 0.0 {
                    let overlay = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        graphics::Rect::new(0.0, 0.0, 320.0, 320.0),
                        Color::new(0.0, 0.0, 0.0, self.fade_alpha),
                    )?;
                    canvas.draw(&overlay, DrawParam::default());
                }

                canvas.finish(ctx)?;
                return Ok(());
            }
            MenuPhase::SplashGameTitle => {
                let mut canvas = Canvas::from_frame(ctx, Color::from_rgb(10, 10, 10));

                let lead_pos = self.block_target_path[self.block_path_index];
                let lead_rect = graphics::Rect::new(lead_pos.x, lead_pos.y, 32.0, 32.0);
                let lead_mesh = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    lead_rect,
                    Color::from_rgb(50, 120, 255),
                )?;
                canvas.draw(&lead_mesh, DrawParam::default());

                if self.block_timer > 3.0 {
                    let t = ((self.block_timer - 3.0) / 4.5).clamp(0.0, 1.0);
                    let alpha = t * t;

                    let scale_factor = 164.0 / 1024.0;
                    let dest = Vec2::new(160.0 - 164.0 / 2.0, 100.0 - 164.0 / 2.0);

                    canvas.draw(
                        &self.fade_image,
                        DrawParam::default()
                            .dest(dest)
                            .scale(Vec2::splat(scale_factor))
                            .color(Color::new(1.0, 1.0, 1.0, alpha)),
                    );
                }

                for p in &self.destruction_particles {
                    let rect = graphics::Rect::new(p.pos.x, p.pos.y, p.size, p.size);
                    let mesh = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        rect,
                        p.color,
                    )?;
                    canvas.draw(&mesh, DrawParam::default());
                }

                let chaser_rect = graphics::Rect::new(
                    self.block_chaser_pos.x,
                    self.block_chaser_pos.y,
                    32.0,
                    32.0,
                );
                let chaser_mesh = graphics::Mesh::new_rectangle(
                    ctx,
                    graphics::DrawMode::fill(),
                    chaser_rect,
                    Color::from_rgb(200, 50, 50),
                )?;
                canvas.draw(&chaser_mesh, DrawParam::default());

                if self.block_timer > 1.0 {
                    let t = ((self.block_timer - 1.0) * 4.0).min(1.0);
                    let bounce = (t * std::f32::consts::PI).sin() * 8.0;
                    let alpha = t;

                    let color = if t < 1.0 {
                        Color::new(1.0, 1.0, 1.0, alpha) // fade-in white
                    } else {
                        let transition = ((self.block_timer - 3.5) / 1.5).clamp(0.0, 1.0);
                        let r = 1.0;
                        let g = 1.0;
                        let b = 1.0 - transition; // white → yellow (removing blue)
                        Color::new(r, g, b, 1.0)
                    };

                    let title =
                        Text::new(TextFragment::new("Chasin' Blocks").scale(36.0).color(color));

                    let dims = title.measure(ctx)?;
                    let x = 160.0 - dims.x / 2.0;
                    let y = 270.0 - bounce;
                    canvas.draw(&title, DrawParam::default().dest(Vec2::new(x, y)));
                }

                if self.block_timer > 6.0 {
                    let fade = ((self.block_timer - 6.0) / 1.5).clamp(0.0, 1.0);
                    let overlay = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        graphics::Rect::new(0.0, 0.0, 320.0, 320.0),
                        Color::new(0.0, 0.0, 0.0, fade),
                    )?;
                    canvas.draw(&overlay, DrawParam::default());
                }

                canvas.finish(ctx)?;
                return Ok(());
            }
            _ => {}
        }
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
            let mesh =
                graphics::Mesh::new_rectangle(ctx, graphics::DrawMode::fill(), rect, p.color)?;
            canvas.draw(&mesh, DrawParam::default());
        }

        if self.showing_credits {
            let lines = vec![
                "Chasin' Blocks",
                "",
                "Developed by: Cardboard Studio",
                "",
                "Graphics",
                "Cardboard Graphics",
                "",
                "Menu Music",
                "'Unforgiving Lands'",
                "by",
                "'HorrorPen'",
                "Licensed under CC-BY 3.0",
                "",
                "In-Game Music",
                "'Driving In The Rain'",
                "by",
                "'onemansymphony'",
                "Licensed under CC-BY 4.0",
                "",
                "Press [Esc] to return",
            ];

            for (i, line) in lines.iter().enumerate() {
                let mut fragment = TextFragment::new((*line).to_string());

                if i == 0 {
                    fragment = fragment.color(Color::YELLOW);
                } else if line.contains("Press [Esc]") {
                    fragment = fragment.color(Color::from_rgb(220, 80, 80));
                } else {
                    fragment = fragment.color(Color::WHITE);
                }

                let text = Text::new(fragment);
                let dims = text.measure(ctx).unwrap();
                let x = screen_w / 2.0 - dims.x / 2.0;
                let y = self.credits_offset + i as f32 * 20.0;
                canvas.draw(&text, DrawParam::default().dest(Vec2::new(x, y)));
            }
        } else {
            let title = graphics::Text::new("Chasin' Blocks");
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
                let dims = text.measure(ctx).unwrap();
                let x = screen_w / 2.0 - dims.x / 2.0;
                let y = 100.0 + i as f32 * 40.0;

                canvas.draw(&text, DrawParam::default().dest(Vec2::new(x, y)));
            }
        }

        canvas.finish(ctx)?;
        Ok(())
    }
}
