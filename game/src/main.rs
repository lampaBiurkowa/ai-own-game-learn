
use dqn_linear_agent::train_agent;
// use ai_raw_agent::train_agent;
// use ggez::{conf::WindowMode, event, GameResult};
use server::run_http_server;
// use ui::Game;

mod transport;
mod engine;
mod server;
// mod ui;
mod client;
mod ai_raw_agent;
mod dqn_linear_agent;

pub fn main()
{
//  -> GameResult {
    // let cb = ggez::ContextBuilder::new("game", "ggez").window_mode(WindowMode::dimensions(WindowMode::default(), 320.0, 320.0)).add_resource_path("./resources");
    // let (mut ctx, events_loop) = cb.build()?;
    std::thread::spawn(move || run_http_server());
    // train_agent("http:localhost:3030", 1, 1000);
    // train_dqn("http:localhost:3030", 1, 1000);
    // let game = Game::new(&mut ctx)?;
    // event::run(ctx, events_loop, game);

    train_agent("http:localhost:3030", 1, 10000);

    // Ok(())
}