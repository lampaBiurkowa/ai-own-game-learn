use ggez::{conf::{WindowMode, WindowSetup}, event, input::keyboard::KeyInput, Context, GameResult};
use server::run_http_server;

mod client;
mod engine;
mod menu;
mod server;
mod transport;
mod ui;

use menu::MenuState;
use ui::Game;

enum AppState {
    Menu(MenuState),
    Game(Game),
}

impl event::EventHandler<ggez::GameError> for AppState {
    fn update(&mut self, ctx: &mut ggez::Context) -> GameResult {
        match self {
            AppState::Menu(state) => state.update(ctx),
            AppState::Game(state) => state.update(ctx),
        }
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> GameResult {
        match self {
            AppState::Menu(state) => state.draw(ctx),
            AppState::Game(state) => state.draw(ctx),
        }
    }

    fn key_down_event(&mut self, ctx: &mut Context, input: KeyInput, repeat: bool) -> GameResult {
        match self {
            AppState::Menu(menu) => {
                if let Some(new_state) = menu.handle_input(ctx, input)? {
                    *self = new_state;
                }
                Ok(())
            }
            AppState::Game(game) => game.key_down_event(ctx, input, repeat),
        }
    }
}

fn main() -> GameResult {
    std::thread::spawn(|| run_http_server());
    let cb = ggez::ContextBuilder::new("game", "ggez")
        .window_setup(WindowSetup::default().title("Chasin' Blocks"))
        .window_mode(WindowMode::dimensions(WindowMode::default(), 320.0, 320.0))
        .add_resource_path("./resources");
    let (ctx, event_loop) = cb.build()?;

    let state = AppState::Menu(MenuState::new());
    event::run(ctx, event_loop, state)
}
