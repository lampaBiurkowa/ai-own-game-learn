use reqwest::blocking::Client;

use crate::{engine::GameState, server::MoveCommand};

#[derive(Clone)]
pub(crate) struct GameApiClient {
    client: Client,
    base_url: String,
    seed: u64
}

impl GameApiClient {
    pub fn new(base_url: &str, seed: u64) -> Self {
        GameApiClient {
            client: Client::new(),
            base_url: base_url.to_string(),
            seed
        }
    }

    pub fn move_player(&self, dx: isize, dy: isize) -> GameState {
        let move_command = MoveCommand { dx, dy };
        let resp = self.client
            .post(format!("{}/game/{}/move", self.base_url, self.seed))
            .json(&move_command)
            .send();

        resp.unwrap().json().unwrap()
    }

    pub fn fetch_state(&self) -> GameState {
        let resp = self.client
            .get(format!("{}/game/{}/state", self.base_url, self.seed))
            .send();

        resp.unwrap().json().unwrap()
    }

    pub fn fetch_initial_state(&self) -> Option<GameState> {
        let resp = self.client
            .get(format!("{}/game/{}/init", self.base_url, self.seed))
            .send();

        resp.unwrap().json().unwrap()
    }
}