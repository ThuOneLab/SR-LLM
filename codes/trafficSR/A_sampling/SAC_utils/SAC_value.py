import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.trafficSR.A_sampling.policy_network.SRmodel import labelEmbedding, newEmbedding
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import UNITS_VECTOR_SIZE
class FCQS_new(nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple = (256, 256),
                 embedding_dim: tuple = (128, 128),
                 n_layers: tuple = (1, 1),
                 overall_and_partial: tuple = (True, True),
                 activation_fc=lambda x: F.relu(x),): #发现应该是F.relu(x)
        super(FCQS_new, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_overall, self.use_partial = overall_and_partial
        self.output_size = action_space_dim

        # --------- Embedding layer ---------
        self.overall_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[0], input_dim=obs_space_dim[0], out_dim=self.hidden_size[0])
        self.partial_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[1], input_dim=3, out_dim=self.hidden_size[1] - (obs_space_dim[1] - 3))
        
        # --------- Partial Onehot Dense layer ---------
        self.partial_onehot_dense = torch.nn.Linear(obs_space_dim[1], self.hidden_size[1])
        
        # # --------- Hidden layer ---------
        overall_hidden_layers, partial_hidden_layers = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.n_layers[0]):
            overall_hidden_layers.append(torch.nn.Linear(self.hidden_size[0], self.hidden_size[0]))
        for _ in range(self.n_layers[1]):
            partial_hidden_layers.append(torch.nn.Linear(self.hidden_size[1], self.hidden_size[1]))
        self.overall_hidden_layers = overall_hidden_layers
        self.partial_hidden_layers = partial_hidden_layers

        # --------- Stacked RNN cells ---------
        # overall_stacked_cells, partial_stacked_cells = None, None
        # overall_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[0],hidden_size=self.hidden_size[0]) for _ in range(self.n_layers[0])])
        # partial_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[1],hidden_size=self.hidden_size[1]) for _ in range(self.n_layers[1])])
        # self.overall_stacked_cells = overall_stacked_cells
        # self.partial_stacked_cells = partial_stacked_cells
        
        # --------- Activation function ---------
        self.activation_fc = activation_fc

        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        output_dense = []
        output_dense.append(torch.nn.Linear(self.hidden_size[0], self.output_size))
        output_dense.append(torch.nn.Linear(self.hidden_size[1], self.output_size))
        self.output_dense = output_dense

        # --------- concat network output ---------
        dim = 2 if (self.use_partial and self.use_overall) else 1
        self.concat_output_layers = torch.nn.Linear(dim * self.output_size, self.output_size)
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)

    def fwd_overall(self, input_tensor):
        # --------- Input dense layer ---------
        x = self.overall_embedding(input_tensor.int())  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        for i in range(self.n_layers[0]):
            x = self.overall_hidden_layers[i](x)  # (batch_size, hidden_size)
            x= self.activation_fc(x)
        # --------- Output dense layer ---------
        x = self.output_dense[0](x)  # (batch_size, output_size)
        # --------- Applying activation function ---------
        x = self.activation_fc(x)  # (batch_size, output_size)
        # --------------- Return ---------------
        return x  # (batch_size, output_size)

    def fwd_partial(self, input_tensor):
        # --------- Input dense layer ---------
        # x = self.partial_embedding(input_tensor[:, :3].int())  # (batch_size, hidden_size)
        # x = torch.cat((x, input_tensor[:, 3:]), dim=1)
        x = self.partial_onehot_dense(input_tensor)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        for i in range(self.n_layers[1]):
            x = self.partial_hidden_layers[i](x)  # (batch_size, hidden_size)
            x= self.activation_fc(x)
        # --------- Output dense layer ---------
        x = self.output_dense[1](x)  # (batch_size, output_size)
        # --------- Applying activation function ---------
        x = self.activation_fc(x)  # (batch_size, output_size)
        # --------------- Return ---------------
        return x  # (batch_size, output_size)

    def forward(self,
                partial_obs,
                overall_obs,
                ):
        if self.use_overall:
            res_overall = self.fwd_overall(overall_obs)
            res_overall_cloned = res_overall.clone()
        if self.use_partial:
            res_partial = self.fwd_partial(partial_obs)
            res_partial_cloned = res_partial.clone()

        if self.use_overall and self.use_partial:
            res_sum = torch.cat([res_overall_cloned, res_partial_cloned, ], dim=1)
            res = self.concat_output_layers(res_sum)
        elif self.use_overall:
            res = self.concat_output_layers(res_overall_cloned)
        elif self.use_partial:
            res = self.concat_output_layers(res_partial_cloned)
        else:
            raise ValueError("No input to the network")
        # res = self.output_activation(res)
        return res

    # useless
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

class FCQS(nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple = (64, 64),
                 embedding_dim: tuple = (32, 32),
                 n_layers: tuple = (2, 2),
                 overall_and_partial: tuple = (True, True),
                 activation_fc=lambda x: F.relu(x),): #发现应该是F.relu(x)
        super(FCQS, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_overall, self.use_partial = overall_and_partial
        self.output_size = action_space_dim

        # --------- Embedding layer ---------
        self.overall_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[0],
                                                input_dim=obs_space_dim[0], out_dim=self.hidden_size[0])
        self.partial_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[1],
                                                input_dim=3, out_dim=self.hidden_size[1] - (obs_space_dim[1] - 3))
        
        # --------- Partial Onehot Dense layer ---------
        self.partial_onehot_dense = torch.nn.Linear(obs_space_dim[1], self.hidden_size[1])
        
        # --------- Hidden layer ---------
        overall_hidden_layers, partial_hidden_layers = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.n_layers[0]):
            overall_hidden_layers.append(torch.nn.Linear(self.hidden_size[0], self.hidden_size[0]))
        for _ in range(self.n_layers[1]):
            partial_hidden_layers.append(torch.nn.Linear(self.hidden_size[1], self.hidden_size[1]))
        self.overall_hidden_layers = overall_hidden_layers
        self.partial_hidden_layers = partial_hidden_layers

        self.activation_fc = activation_fc

        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        output_dense = []
        output_dense.append(torch.nn.Linear(self.hidden_size[0], self.output_size))
        output_dense.append(torch.nn.Linear(self.hidden_size[1], self.output_size))
        self.output_dense = output_dense

        # --------- concat network output ---------
        dim = 2 if (self.use_partial and self.use_overall) else 1
        self.concat_output_layers = torch.nn.Linear(dim * self.output_size,
                                                    self.output_size)

    def fwd_overall(self, input_tensor):
        # --------- Input dense layer ---------
        x = self.overall_embedding(input_tensor.int())  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        for i in range(self.n_layers[0]):
            x = self.overall_hidden_layers[i](x)  # (batch_size, hidden_size)
        # --------- Output dense layer ---------
        x = self.output_dense[0](x)  # (batch_size, output_size)
        # --------- Applying activation function ---------
        x = self.activation_fc(x)  # (batch_size, output_size)
        # --------------- Return ---------------
        return x  # (batch_size, output_size)

    def fwd_partial(self, input_tensor):
        # --------- Input dense layer ---------
        # x = self.partial_embedding(input_tensor[:, :3].int())  # (batch_size, hidden_size)
        # x = torch.cat((x, input_tensor[:, 3:]), dim=1)
        x = self.partial_onehot_dense(input_tensor)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        for i in range(self.n_layers[1]):
            x = self.partial_hidden_layers[i](x)  # (batch_size, hidden_size)
        # --------- Output dense layer ---------
        x = self.output_dense[1](x)  # (batch_size, output_size)
        # --------- Applying activation function ---------
        x = self.activation_fc(x)  # (batch_size, output_size)
        # --------------- Return ---------------
        return x  # (batch_size, output_size)

    def forward(self,
                partial_obs,
                overall_obs,
                ):
        if self.use_overall:
            res_overall = self.fwd_overall(overall_obs)
            # res_overall_cloned = res_overall.clone()
        if self.use_partial:
            res_partial = self.fwd_partial(partial_obs)
            # res_partial_cloned = res_partial.clone()

        if self.use_overall and self.use_partial:
            res_sum = torch.cat([res_overall, res_partial, ], dim=1)
            res = self.concat_output_layers(res_sum)
        elif self.use_overall:
            res = self.concat_output_layers(res_overall)
        elif self.use_partial:
            res = self.concat_output_layers(res_partial)
        else:
            raise ValueError("No input to the network")
        res = self.activation_fc(res)
        return res

    # useless
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class ValueNetwork():
    def __init__(self, train_args, action_space_dim, device, dtype):
        self.model = None
        self.train_args = train_args
        self.device = device
        self.dtype = dtype
        self.init_model(train_args, action_space_dim)

    def init_model(self, train_args, action_space_dim):
        self.model = FCQS_new(
            obs_space_dim=(train_args['env_args']['max_time_step'],
                           3*action_space_dim + 4 * (UNITS_VECTOR_SIZE + 1) + 1),
            action_space_dim=action_space_dim,
            **train_args['sac_value_args']
        ).to(self.device)
