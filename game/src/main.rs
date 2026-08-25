use ggez::{conf::WindowMode, event, GameResult};
use server::run_http_server;
use ui::Game;

mod achievements;
mod client;
mod engine;
mod server;
mod transport;
mod ui;

// The training experiments. Not part of the shipped client; switching either back on
// means putting `burn` back in Cargo.toml too.
// mod ai_raw_agent;
// mod dqn_linear_agent;

pub fn main() -> GameResult {
    // Started before the window, so the local game server is already listening by the
    // time GameClient asks it for the initial state.
    std::thread::spawn(run_http_server);
    achievements::init();

    let cb = ggez::ContextBuilder::new("game", "ggez")
        .window_mode(WindowMode::dimensions(WindowMode::default(), 320.0, 320.0))
        .add_resource_path("./resources");
    let (mut ctx, events_loop) = cb.build()?;
    let game = Game::new(&mut ctx)?;
    event::run(ctx, events_loop, game)
}
