import torch
from codes.trafficSR.A_sampling.policy_network.SRmodel import get_model
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import UNITS_VECTOR_SIZE


class PolicyNetwork():
    def __init__(self, train_args, action_space_dim, device, dtype):
        self.model = None
        self.train_args = train_args
        self.device = device
        self.dtype = dtype
        self.model_type = self.train_args['model_type']
        self.init_model(train_args, action_space_dim)
        self.overall_partial_models=['lstm','embedding_lstm','combined_lstm']

    def init_model(self, train_args, action_space_dim):
        if train_args['model_type'] == "overall_lstm":
            self.model = get_model(
                model_type=train_args['model_type'],
                obs_space_dim=train_args['env_args']['max_time_step'],
                action_space_dim=action_space_dim,
                device=self.device,
                **train_args['model_args']
            ).to(self.device)
        elif train_args['model_type'] == "lstm":
            self.model = get_model(
                model_type=train_args['model_type'],
                obs_space_dim=(train_args['env_args']['max_time_step'],
                               3 * action_space_dim + 4 * (UNITS_VECTOR_SIZE + 1) + 1),
                action_space_dim=action_space_dim,
                device=self.device,
                **train_args['model_args']
            ).to(self.device)
        elif train_args['model_type'] =="embedding_lstm":
            self.model = get_model(
                model_type=train_args['model_type'],
                obs_space_dim=(train_args['env_args']['max_time_step'],
                               3 + 4 * (UNITS_VECTOR_SIZE + 1) + 1),
                action_space_dim=action_space_dim,
                device=self.device,
                **train_args['model_args']
            ).to(self.device)
        elif train_args['model_type'] =="combined_lstm":
            self.model = get_model(
                model_type=train_args['model_type'],
                obs_space_dim=(train_args['env_args']['max_time_step'],
                               3*action_space_dim + 4 * (UNITS_VECTOR_SIZE + 1) + 1),
                action_space_dim=action_space_dim,
                device=self.device,
                **train_args['model_args']
            ).to(self.device)

    def get_observations(self, env):
        if self.model_type == 'overall_lstm':
            observations = env.get_observation()
            observations = torch.tensor(observations).to(self.device, self.dtype)  # (batch_size, output_size
        elif self.model_type in self.overall_partial_models:
            overall_observation = env.get_overall_observation()
            overall_observation = torch.tensor(overall_observation).to(self.device,self.dtype)
            partial_observation = env.get_partial_observation()
            partial_observation = torch.tensor(partial_observation).to(self.device,self.dtype)
            observations = (overall_observation, partial_observation)
        elif self.model_type == 'transformer':
            observations = env.get_observation()
            observations = torch.tensor(observations).to(self.device, self.dtype)  # (batch_size, output_size
        return observations

    def output_and_state(self, states, states_partial, states_overall, time_id, batch_size, observations):
        output = None
        if self.model_type == 'overall_lstm':
            output, states = self.model(
                input_tensor=observations,  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)
                states=states
            )
        elif self.model_type in self.overall_partial_models:
            output, states_overall, states_partial, = self.model(
                partial_obs=observations[1],
                overall_obs=observations[0],
                states_partial=states_partial,
                states_overall=states_overall,
            )
        elif self.model_type == 'transformer':
            output = self.model(
                obs=observations,
                time=torch.ones((batch_size,), dtype=torch.int64) * time_id,
            )
        return output, states, states_partial, states_overall
