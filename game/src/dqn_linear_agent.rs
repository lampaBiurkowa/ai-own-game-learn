use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::nn::conv::Conv2d;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::PaddingConfig2d;
use burn::nn::{conv::Conv2dConfig, Linear, LinearConfig};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{BasicOps, TensorData};
use burn::tensor::{activation::relu, Tensor};
use burn::module::Module;
use burn::train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep};
use rand::Rng;
use rand::prelude::*;
use std::collections::VecDeque;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

use crate::client::GameClient;
use crate::engine::{Cell, GameOverCause};

const DISCOUNT: f32 = 0.99;
const REPLAY_MEMORY_SIZE: usize = 1000;
const MIN_REPLAY_MEMORY_SIZE: usize = 100;
const MINIBATCH_SIZE: usize = 64;
const EPSILON_DECAY: f32 = 0.995;
const MIN_EPSILON: f32 = 0.01;
const UPDATE_TARGET_EVERY: usize = 10;

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = "4")]
    num_actions: usize,
    #[config(default = "128")]
    hidden_size: usize,
    #[config(default = "3")]
    kernel_size: usize,
    #[config(default = "5")]
    field_types: usize,
    #[config(default = "10")]
    viewport_size: usize,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DQNModel<B> {
        let conv1 = Conv2dConfig::new([self.field_types, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let fc1 = LinearConfig::new(1600, 128).init(device);
        let fc2 = LinearConfig::new(128, self.num_actions).init(device);

        DQNModel { conv1, fc1, fc2 }
    }
}

#[derive(Module, Debug)]
struct DQNModel<B: Backend> {
    conv1: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> DQNModel<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = relu(x);
        let x = x.flatten(1, 3);
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.fc2.forward(x);
        x
    }

    fn forward_regression(&self, batch: StateBatch<B>) -> RegressionOutput<B> {
        let predicted_q_values = self.forward(batch.grids.clone());
        let loss = MseLoss::new().forward(predicted_q_values.clone(), batch.q_values.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output: predicted_q_values,
            targets: batch.q_values,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<StateBatch<B>, RegressionOutput<B>> for DQNModel<B> {
    fn step(&self, item: StateBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<StateBatch<B>, RegressionOutput<B>> for DQNModel<B> {
    fn step(&self, item: StateBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item)
    }
}

#[derive(Clone, Debug)]
struct StateItem {
    grid_flattened: Vec<f32>,
    q_values: Vec<f32>
}

#[derive(Clone, Debug)]
pub struct StateBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct StateBatch<B: Backend> {
    pub grids: Tensor<B, 4>,
    pub q_values: Tensor<B, 2>,
}

impl<B: Backend> StateBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<StateItem, StateBatch<B>> for StateBatcher<B> {
    fn batch(&self, items: Vec<StateItem>) -> StateBatch<B> {
        let batch_size = items.len();  // use the actual batch size

        let items_flattened = items
            .iter()
            .map(|x| &x.grid_flattened)
            .flatten()
            .cloned()
            .collect::<Vec<f32>>();

        let grids = Tensor::<B, 1>::from_data(TensorData::from(items_flattened.as_slice()), &self.device)
            .reshape([batch_size, 5, 10, 10]);

        let q_values_flattened = items
            .iter()
            .map(|x| &x.q_values)
            .flatten()
            .cloned()
            .collect::<Vec<f32>>();

        let q_values = Tensor::<B, 1>::from_data(TensorData::from(q_values_flattened.as_slice()), &self.device)
            .reshape([batch_size, 4]);

        StateBatch { grids, q_values }
    }
}

struct StatesDataset {
    items: InMemDataset<StateItem>
}

impl StatesDataset {
    fn new(items: InMemDataset<StateItem>) -> Self {
        Self { items }
    }
}

impl Dataset<StateItem> for StatesDataset {
    fn get(&self, index: usize) -> Option<StateItem> {
        self.items.get(index)
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

pub(crate) fn train_agent(url: &str, seed: u64, episodes: usize) {
    let device = WgpuDevice::IntegratedGpu(0);
    let mut model: DQNModel<Autodiff<Wgpu>> = ModelConfig::new().init(&device);
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    // model = model.clone().load_file("30-12--3-10000", &recorder, &device).unwrap();
    let mut target_model: DQNModel<Autodiff<Wgpu>> = ModelConfig::new().init(&device);
    let mut replay_memory = VecDeque::<(Vec<f32>, i32, f32, Vec<f32>, bool)>::with_capacity(REPLAY_MEMORY_SIZE);
    let mut target_update_counter = 0;
    let batcher = StateBatcher::new(device.clone());
    let valid_batcher = StateBatcher::<Wgpu>::new(device.clone());
    let mut epsilon = 1.0;
    let optimizer_config = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-3)));
    let mut episode_scores: Vec<usize> = vec![];
    let mut episode_rewards: Vec<f32> = vec![];
    for episode in 1..=episodes {
        let mut episode_reward = 0.0;
        let mut game_client = GameClient::new(url, seed + episode as u64);
        let mut state = game_client.get_grid();
        let mut done = false;
        let mut step = 0;
        let mut no_progress_moves = 0;
        let mut illegal_moves = 0;
        let mut rightwards_moves = 0;
        let mut enemies_killed = 0;

        while !done {


            let grid_flattened = state
                .clone()
                .into_iter()
                .flatten()
                .map(|x| (0..=4).map(|i| match x {
                    Cell::Floor if i == 0 => 1.0,
                    Cell::Obstacle if i == 1 => 1.0,
                    Cell::Player if i == 2 => 1.0,
                    Cell::EnemyVertical(_) if i == 3 => 1.0,
                    Cell::EnemyHorizontal(_) if i == 4 => 1.0,
                    _ => 0.0
                }).collect::<Vec<f32>>())
                .flatten()
                .collect::<Vec<f32>>();
            let mut action = rand::thread_rng().gen_range(0..=3);
            if rand::random::<f32>() < epsilon {
                match action {
                    0 => game_client.move_player(0, 1),
                    1 => game_client.move_player(1, 0),
                    2 => game_client.move_player(0, -1),
                    3 => game_client.move_player(-1, 0),
                    _ => ()
                }
            } else {
                let output = model.clone().forward(
                    Tensor::<Autodiff<Wgpu>, 1>::from_data(
                        TensorData::from(
                            grid_flattened.as_slice()), &device
                        ).reshape([1, 5, 10, 10])
                    );
                let q_values = tensor_to_vec(output.clone());
                let x = q_values[0].iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index).unwrap();
                
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("actions.log").unwrap();
                writeln!(
                    file,
                    "0: {}, 1: {}, 2:{} 3: {} took: {}",
                    q_values[0][0], q_values[0][1], q_values[0][2], q_values[0][3], x
                ).unwrap();

                action = x as i32;
                match action {
                    0 => game_client.move_player(0, 1),
                    1 => game_client.move_player(1, 0),
                    2 => game_client.move_player(0, -1),
                    3 => game_client.move_player(-1, 0),
                    _ => ()
                }
            };

            if game_client.is_game_over().is_some() {
                done = true;
            }

            let mut reward = match game_client.is_game_over() {
                Some(GameOverCause::Enemy) => -0.0,
                _ => 0.0
            };

            if !done && !game_client.player_moved() {
                reward -= 5.0;
                illegal_moves += 1;
                done = true;
            }

            reward += game_client.get_player_position().0 as f32 / 10.0;
            reward += if game_client.score_increased() {
                0.0//game_client.get_score() as f32 // 10.0
            } else if game_client.player_moved_rightwards() {
                rightwards_moves += 1;
                0.0
            } else if game_client.player_moved() {
                no_progress_moves += 1;
                -0.0
            } else {
                0.0
            };

            if game_client.enemy_killed() {
                enemies_killed += 1;
            }

            let new_state = game_client.get_grid();
            let new_grid_flattened = new_state
                .clone()
                .into_iter()
                .flatten()
                .map(|x| (0..=4).map(|i| match x {
                    Cell::Floor if i == 0 => 1.0,
                    Cell::Obstacle if i == 1 => 1.0,
                    Cell::Player if i == 2 => 1.0,
                    Cell::EnemyVertical(_) if i == 3 => 1.0,
                    Cell::EnemyHorizontal(_) if i == 4 => 1.0,
                    _ => 0.0
                }).collect::<Vec<f32>>())
                .flatten()
                .collect::<Vec<f32>>();

            episode_reward += reward;
            replay_memory.push_back((grid_flattened, action, reward, new_grid_flattened, done));
            if replay_memory.len() > REPLAY_MEMORY_SIZE {
                replay_memory.pop_front();
            }

            if replay_memory.len() > MIN_REPLAY_MEMORY_SIZE {

                let minibatch = replay_memory.clone().into_iter().choose_multiple(&mut rand::thread_rng(), MINIBATCH_SIZE);

                let current_states = minibatch.clone().into_iter().map(|x| x.0).collect::<Vec<_>>();
                let current_states_flattened = current_states
                    .clone()
                    .into_iter()
                    .flatten().collect::<Vec<f32>>();
                
                let current_qs_list = model.forward(
                    Tensor::<Autodiff<Wgpu>, 1>::from_data(
                        TensorData::from(current_states_flattened.as_slice()), &device)
                            .reshape([current_states.len(), 5, 10, 10]
                    )
                );

                let new_current_states = minibatch.clone().into_iter().map(|x| x.3).collect::<Vec<_>>();
                let new_current_states_flattened = new_current_states.clone().into_iter().flatten().collect::<Vec<f32>>();
                let new_qs_list = target_model.forward(
                    Tensor::<Autodiff<Wgpu>, 1>::from_data(
                        TensorData::from(new_current_states_flattened.as_slice()), &device)
                            .reshape([new_current_states.len(), 5, 10, 10]
                    )
                );

                let new_q_values_vec = tensor_to_vec(new_qs_list);
                let current_q_values_vec = tensor_to_vec(current_qs_list);

                let mut items = vec![];
                for (index, new_q_values) in new_q_values_vec.iter().enumerate() {
                    let (state, action, reward, new_state, done_inner) = &minibatch[index];
                    let new_q = match done_inner {
                        true => *reward,
                        false => {
                            let max_future_q = new_q_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                            reward + DISCOUNT * max_future_q
                        }
                    };

                    let mut current_qs = current_q_values_vec[index].clone();
                    current_qs[*action as usize] = new_q;
                    items.push(StateItem { grid_flattened: state.clone(), q_values: current_qs });
                }

                let dataset = StatesDataset::new(InMemDataset::new(items));

                let dataloader_train = DataLoaderBuilder::new(batcher.clone())
                    .batch_size(MINIBATCH_SIZE)
                    .num_workers(1)
                    .build(dataset);

                let learner = 
                LearnerBuilder::new("artifacts")
                    .devices(vec![device.clone()])
                    .num_epochs(1)
                    .build(model.clone(), optimizer_config.init(), 1e-4);
                model = learner.fit(dataloader_train, DataLoaderBuilder::new(valid_batcher.clone()).batch_size(MINIBATCH_SIZE).build(InMemDataset::<StateItem>::new(vec![])));
                if done {
                    target_update_counter += 1;
                }
                if target_update_counter > UPDATE_TARGET_EVERY {
                    let path = env::temp_dir().join("dqn-test");
                    model.clone().save_file(path.clone(), &recorder).unwrap();
                    target_model = target_model.clone().load_file(path, &recorder, &device).unwrap();
                    target_update_counter = 0;
                }
                if episode % 200 == 0 {
                    model.clone().save_file(format!("04-01--{}", episode), &recorder).unwrap();
                }
            }

            state = new_state;
            step += 1;
            if step > 150 {
                break;
            }
        }

        if episode > MIN_REPLAY_MEMORY_SIZE && epsilon > MIN_EPSILON {
            epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);
        }
        episode_scores.push(game_client.get_score());
        episode_rewards.push(episode_reward);
        let average_score = episode_scores.iter().sum::<usize>() as f32 / episode_scores.len() as f32;
        let average_reward = episode_rewards.iter().sum::<f32>() as f32 / episode_rewards.len() as f32;
        let last_scores = episode_scores.iter().rev().take(50).collect::<Vec<_>>().into_iter().rev().cloned().collect::<Vec<_>>();
        let last_average_scores = last_scores.iter().sum::<usize>() as f32 / last_scores.len() as f32;
        let last_rewards = episode_rewards.iter().rev().take(50).collect::<Vec<_>>().into_iter().rev().cloned().collect::<Vec<_>>();
        let last_average_rewards = last_rewards.iter().sum::<f32>() as f32 / last_scores.len() as f32;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("episode_rewards.log").unwrap();
        writeln!(
            file,
            "Episode: {} (eps: {}), Reward: {}, score:{} bad moves: {} enemies: {} illegal {} rightward {}, avg reward: {}, last avg reward: {}, Avg score: {}, Last Avg score: {}",
            episode, epsilon, episode_reward, game_client.get_score(), no_progress_moves, enemies_killed, illegal_moves, rightwards_moves, average_reward, last_average_rewards, average_score, last_average_scores
        ).unwrap();
    }
}

fn tensor_to_vec<B: Backend, K: BasicOps<B>>(tensor: Tensor<B, 2, K>) -> Vec<Vec<f32>> {
    let (x, y) = (tensor.dims()[0], tensor.dims()[1]);
    let data = tensor.flatten::<1>(0, 1).into_data().to_vec::<f32>().unwrap();
    let mut result = Vec::with_capacity(x);
    for i in 0..x {
        let row = data.as_slice().iter().skip(i * y).take(y).map(|x| *x).collect::<Vec<_>>();
        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {

    use super::*;
    use burn::tensor::Tensor;
    #[test]
    fn test_2x3_tensor() {
        let tensor = Tensor::<Wgpu, 1>::from_data(
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).as_slice(),
            &WgpuDevice::IntegratedGpu(0),
        ).reshape([2, 3]);
        let result = tensor_to_vec(tensor);
        assert_eq!(result, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }
    
    #[test]
    fn test_1x5_tensor() {
        let tensor = Tensor::<Wgpu, 1>::from_data(
            (vec![1.0, 2.0, 3.0, 4.0, 5.0]).as_slice(),
            &WgpuDevice::IntegratedGpu(0),
        ).reshape([1, 5]);
        let result = tensor_to_vec(tensor);
        assert_eq!(result, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
    }
    
    #[test]
    fn test_3x1_tensor() {
        let tensor = Tensor::<Wgpu, 1>::from_data(
            (vec![1.0, 2.0, 3.0]).as_slice(),
            &WgpuDevice::IntegratedGpu(0),
        ).reshape([3, 1]);
        let result = tensor_to_vec(tensor);
        assert_eq!(result, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }
    
    #[test]
    fn test_1x1_tensor() {
        let tensor = Tensor::<Wgpu, 1>::from_data(
            (vec![42.0]).as_slice(),
            &WgpuDevice::IntegratedGpu(0),
        ).reshape([1, 1]);
        let result = tensor_to_vec(tensor);
        assert_eq!(result, vec![vec![42.0]]);
    }
}






            // let grid_flattened1 = state
            //     .clone()
            //     .into_iter()
            //     .flatten()
            //     .map(|x| (0..=4).map(|i| match x {
            //         Cell::Floor if i == 0 => 1.0,
            //         Cell::Obstacle if i == 1 => 1.0,
            //         Cell::Player if i == 2 => 1.0,
            //         Cell::EnemyVertical(_) if i == 3 => 1.0,
            //         Cell::EnemyHorizontal(_) if i == 4 => 1.0,
            //         _ => 0.0
            //     }).collect::<Vec<f32>>())
            //     .collect::<Vec<Vec<f32>>>();

            // for row in grid_flattened1.chunks(10) {
            //     let row_str: String = row
            //         .iter()
            //         .map(|cell| {
            //             if cell[0] == 1.0 {
            //                 '_'
            //             } else if cell[1] == 1.0 {
            //                 'X'
            //             } else if cell[2] == 1.0 {
            //                 '*'
            //             } else if cell[3] == 1.0 {
            //                 '&'
            //             } else if cell[4] == 1.0 {
            //                 '='
            //             } else {
            //                 ' '
            //             }
            //         })
            //         .collect();
            //     println!("{}", row_str);
            // }