
use ggez::{conf::WindowMode, event, GameResult};
use server::run_http_server;
use ui::Game;

mod transport;
mod engine;
mod server;
mod ui;
mod client;

pub fn main() -> GameResult {
    let cb = ggez::ContextBuilder::new("game", "ggez").window_mode(WindowMode::dimensions(WindowMode::default(), 320.0, 320.0)).add_resource_path("./resources");
    let (mut ctx, events_loop) = cb.build()?;
    std::thread::spawn(move || run_http_server());
    let game = Game::new(&mut ctx)?;
    event::run(ctx, events_loop, game)
}