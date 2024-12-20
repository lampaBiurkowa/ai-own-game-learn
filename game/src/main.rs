
use ggez::{event, GameResult,};
use server::run_http_server;
use ui::Game;

mod client;
mod engine;
mod server;
mod ui;

pub fn main() -> GameResult {
    let cb = ggez::ContextBuilder::new("game", "ggez").add_resource_path("./resources");
    let (mut ctx, events_loop) = cb.build()?;
    std::thread::spawn(move || run_http_server());
    let game = Game::new(&mut ctx)?;
    event::run(ctx, events_loop, game)
}