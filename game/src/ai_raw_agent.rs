// use rand::{thread_rng, Rng};
// use std::collections::HashMap;
// use std::time::Instant;

// use crate::client::GameClient;

// type State = (usize, usize); // Player's position
// type Action = (isize, isize); // Move direction (dx, dy)

// pub(crate) struct QLearningAgent {
//     q_table: HashMap<(State, Action), f64>,
//     learning_rate: f64,
//     discount_factor: f64,
//     exploration_rate: f64,
//     exploration_decay: f64,
//     actions: Vec<Action>,
// }

// impl QLearningAgent {
//     pub(crate) fn new(learning_rate: f64, discount_factor: f64, exploration_rate: f64, exploration_decay: f64) -> Self {
//         let actions = vec![
//             (0, 1),  // Move right
//             (1, 0),  // Move down
//             (-1, 0), // Move up
//             (0, -1), // Move left
//         ];
//         QLearningAgent {
//             q_table: HashMap::new(),
//             learning_rate,
//             discount_factor,
//             exploration_rate,
//             exploration_decay,
//             actions,
//         }
//     }

//     pub(crate) fn choose_action(&mut self, state: State) -> Action {
//         let mut rng = thread_rng();
//         if rng.gen::<f64>() < self.exploration_rate {
//             // Explore: choose a random action
//             self.actions[rng.gen_range(0..self.actions.len())]
//         } else {
//             // Exploit: choose the best action
//             self.actions
//                 .iter()
//                 .cloned()
//                 .max_by(|&a, &b| {
//                     self.q_table
//                         .get(&(state, a))
//                         .unwrap_or(&0.0)
//                         .partial_cmp(self.q_table.get(&(state, b)).unwrap_or(&0.0))
//                         .unwrap()
//                 })
//                 .unwrap_or((0, 1)) // Default action if Q-table is empty
//         }
//     }

//     pub(crate) fn update_q_value(&mut self, state: State, action: Action, reward: f64, next_state: State) {
//         let current_q = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
//         let max_next_q = self
//             .actions
//             .iter()
//             .map(|&a| *self.q_table.get(&(next_state, a)).unwrap_or(&0.0))
//             .fold(f64::MIN, f64::max);

//         let new_q = current_q
//             + self.learning_rate
//                 * (reward + self.discount_factor * max_next_q - current_q);
//         self.q_table.insert((state, action), new_q);
//     }

//     pub(crate) fn decay_exploration_rate(&mut self) {
//         self.exploration_rate *= self.exploration_decay;
//     }
// }

// pub(crate) fn train_agent(url: &str, seed: u64, episodes: usize) {
//     let mut agent = QLearningAgent::new(0.001, 0.95, 0.1, 0.995); // Adjust parameters as needed

//     let mut total_score = 0;
//     let mut last_n= vec![];
//     let n = 50;
//     for episode in 0..episodes {
//         let mut game_client = GameClient::new(url, seed + episode as u64);
//         let mut total_reward = 0.0;
//         let start_time = Instant::now();

//         while !game_client.is_game_over() {
//             if start_time.elapsed().as_secs() >= 30 {
//                 println!("Episode {} timed out.", episode + 1);
//                 break;
//             }

//             let state = game_client.get_player_position();
//             let action = agent.choose_action(state);

//             game_client.move_player(action.0, action.1);

//             let mut reward = if game_client.is_game_over() {
//                 -100.0
//             } else if game_client.score_increased() {
//                 (game_client.get_score() as f64 * game_client.get_score() as f64 * game_client.get_score() as f64) / 1.0
//             } else {
//                 0.0
//             };

//             if game_client.enemy_killed() {
//                 reward += 1.0;
//             }

//             let next_state = game_client.get_player_position();
//             agent.update_q_value(state, action, reward, next_state);
//             total_reward += reward;
//         }

//         let score = game_client.get_score();
//         last_n.push(score);
//         if last_n.len() > n {
//             last_n.remove(0);
//         }
//         total_score += score;
//         let last_n_average = last_n.iter().sum::<usize>() as f32 / n as f32;
//         println!("Episode {}: Total Reward = {}, Score: {}, average: {}. last {} average: {}", episode + 1, total_reward, score, total_score as f32 / episode as f32, n, last_n_average);
//         agent.decay_exploration_rate();
//     }
// }
