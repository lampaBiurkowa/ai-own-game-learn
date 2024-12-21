use std::{collections::HashMap, sync::{Arc, Mutex}};

use serde::{Deserialize, Serialize};
use warp::Filter;

use crate::engine::GameState;

#[derive(Deserialize, Serialize)]
pub(crate) struct MoveCommand {
    pub(crate) dx: isize,
    pub(crate) dy: isize,
}

#[tokio::main]
pub(crate) async fn run_http_server() {

    let games = Arc::new(Mutex::new(HashMap::<u64, GameState>::new()));
    
    let games_clone = Arc::clone(&games);
    let init = warp::path("game")
        .and(warp::path::param::<u64>())
        .and(warp::path("init"))
        .and(warp::get())
        .map(move |seed| {
            let state = GameState::new_with_seed(seed);
            games_clone.lock().unwrap().insert(seed, state.clone());
            warp::reply::json(&state)
        });

    let games_clone = Arc::clone(&games);
    let move_player = warp::path("game")
        .and(warp::path::param::<u64>())
        .and(warp::path("move"))
        .and(warp::post())
        .and(warp::body::json())
        .map(move |seed: u64, cmd: MoveCommand| {
            let mut game = games_clone.lock().unwrap();
            let game = game.get_mut(&seed).unwrap();
            game.move_player(cmd.dx, cmd.dy);
            game.extend_map();
            warp::reply::json(&game)
        });

    let games_clone = Arc::clone(&games);
    let get_state = warp::path("game")
        .and(warp::path::param::<u64>())
        .and(warp::path("state"))
        .and(warp::get())
        .map(move |seed: u64| {
            let game = &games_clone.lock().unwrap()[&seed];
            warp::reply::json(&game)
        });

    let routes = init.or(move_player).or(get_state);
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
