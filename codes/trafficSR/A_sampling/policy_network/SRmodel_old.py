import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from codes.trafficSR.A_sampling.policy_network.SRmodel_utils import labelEmbedding
class CombinedLstmPolicyNetworkOld(torch.nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple = (128, 128),
                 embedding_dim: tuple = (6, 6),
                 n_layers: tuple = (1, 1),
                 overall_and_partial: tuple = (True, True),
                 stacked_cells=None,
                 output_dense=None,
                 device="cuda:1",
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_overall, self.use_partial = overall_and_partial
        # --------- Dense layer ---------
        self.overall_input_dense = torch.nn.Linear(obs_space_dim[0], self.hidden_size[0])
        self.partial_input_dense = torch.nn.Linear(obs_space_dim[1], self.hidden_size[1])
        # --------- Embedding layer ---------
        self.partial_embedding = labelEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[1], input_dim=3, out_dim=self.hidden_size[1] - (obs_space_dim[1] - 3))
        
        # --------- Stacked RNN cells ---------
        self.n_layers = n_layers
        overall_stacked_cells, partial_stacked_cells = None, None
        if stacked_cells is None:
            overall_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[0],
                                                                           hidden_size=self.hidden_size[0])
                                                         for _ in range(self.n_layers[0])])
            partial_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[1],
                                                                           hidden_size=self.hidden_size[1])
                                                         for _ in range(self.n_layers[1])])
        self.overall_stacked_cells = overall_stacked_cells
        self.partial_stacked_cells = partial_stacked_cells
        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        if output_dense is None:
            output_dense = []
            output_dense.append(torch.nn.Linear(self.hidden_size[0], self.output_size).to(device))
            output_dense.append(torch.nn.Linear(self.hidden_size[1], self.output_size).to(device))
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x)
        # --------- concat network output ---------
        dim = 2 if (self.use_partial and self.use_overall) else 1
        self.concat_output_layers = torch.nn.Linear(dim * self.output_size,
                                                    self.output_size)
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)
    def get_zeros_initial_state(self, batch_size, overall_or_partial=0):
        zeros_initial_state = torch.zeros(self.n_layers[overall_or_partial], 2, batch_size,
                                          self.hidden_size[overall_or_partial], requires_grad=False, )
        return zeros_initial_state
    def fwd_overall(self, input_tensor, states):
        # --------- Input dense layer ---------
        # hx = self.overall_embedding(input_tensor.int())  # (batch_size, hidden_size)
        hx = self.overall_input_dense(input_tensor)  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[0]):
            hx, cx = self.overall_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (n_layers, 2, batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (n_layers, 2, batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = torch.add(self.output_dense[0](hx), self.logTemperature)  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)
    
    def fwd_partial(self, input_tensor, states):
        # --------- Input dense layer ---------
        hx1 = self.partial_embedding(input_tensor[:, :3].int())
        hx = torch.cat((hx, input_tensor[:, 3:]), dim=1)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[1]):
            hx, cx = self.partial_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = torch.add(self.output_dense[1](hx), self.logTemperature)  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)
    
    def forward(self,
                partial_obs,
                overall_obs,
                states_partial,
                states_overall,
                ):
        if self.use_overall:
            res_overall, states_overall = self.fwd_overall(overall_obs, states_overall)
            res_overall_cloned = res_overall.clone()
        if self.use_partial:
            res_partial, states_partial = self.fwd_partial(partial_obs, states_partial)
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
        return res, states_overall, states_partial

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params

class SingleLstmPolicyNetwork(torch.nn.Module):
    def __init__(self,
                 obs_space_dim: int,
                 action_space_dim,
                 hidden_size: int,
                 n_layers: int = 1,
                 input_dense=None,
                 stacked_cells=None,
                 output_dense=None,
                 is_lobotomized=False,
                 device="cuda:1",
                 ):
        super().__init__()
        # --------- Input dense layer ---------
        self.input_size = obs_space_dim
        self.hidden_size = hidden_size
        if input_dense is None:
            input_dense = torch.nn.Linear(self.input_size, self.hidden_size)
        self.input_dense = input_dense
        # --------- Stacked RNN cells ---------
        self.n_layers = n_layers
        if stacked_cells is None:
            stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size,
                                                                   hidden_size=self.hidden_size)
                                                 for _ in range(self.n_layers)])
        self.stacked_cells = stacked_cells
        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        if output_dense is None:
            output_dense = torch.nn.Linear(self.hidden_size, self.output_size).to(device)
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x)  # Mapping output to log(p)
        # lambda x: torch.nn.functional.softmax(x, dim=1)
        # torch.sigmoid
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)
        # --------- Lobotomization ---------
        self.is_lobotomized = is_lobotomized

    def get_zeros_initial_state(self, batch_size):
        zeros_initial_state = torch.zeros(self.n_layers, 2, batch_size,
                                          self.hidden_size, requires_grad=False, )
        return zeros_initial_state

    def forward(self,
                input_tensor,  # (batch_size, input_size)
                states,  # (n_layers, 2, batch_size, hidden_size)
                ):
        # --------- Input dense layer ---------
        hx = self.input_dense(input_tensor)  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers):
            hx, cx = self.stacked_cells[i](hx,  # (batch_size, hidden_size)
                                           (states[i, 0, :, :],  # (batch_size, hidden_size)
                                            states[i, 1, :, :]  # (batch_size, hidden_size)
                                            ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense(hx) + self.logTemperature  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params


class LstmPolicyNetwork(torch.nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple,
                 n_layers: tuple = (1, 1),
                 input_dense=None,
                 stacked_cells=None,
                 output_dense=None,
                 is_lobotomized=False,
                 device="cuda:1",
                 ):
        super().__init__()
        # --------- Input dense layer ---------
        # self.input_size = obs_space_dim
        # if input_dense is None:
        #     input_dense = torch.nn.Linear(self.input_size, self.hidden_size)
        self.hidden_size = hidden_size
        self.input_dense = input_dense
        self.overall_input_dense = torch.nn.Linear(obs_space_dim[0], self.hidden_size[0])
        self.partial_input_dense = torch.nn.Linear(obs_space_dim[1], self.hidden_size[1])
        # --------- Stacked RNN cells ---------
        self.n_layers = n_layers
        overall_stacked_cells, partial_stacked_cells = None, None
        if stacked_cells is None:
            overall_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[0],
                                                                           hidden_size=self.hidden_size[0])
                                                         for _ in range(self.n_layers[0])])
            partial_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[1],
                                                                           hidden_size=self.hidden_size[1])
                                                         for _ in range(self.n_layers[1])])
        self.overall_stacked_cells = overall_stacked_cells
        self.partial_stacked_cells = partial_stacked_cells
        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        if output_dense is None:
            output_dense = []
            output_dense.append(torch.nn.Linear(self.hidden_size[0], self.output_size).to(device))
            output_dense.append(torch.nn.Linear(self.hidden_size[1], self.output_size).to(device))
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x)  # Mapping output to log(p)
        # lambda x: torch.nn.functional.softmax(x, dim=1)
        # torch.sigmoid
        # --------- concat network output ---------
        self.concat_output_layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.output_size, self.output_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.output_size, self.output_size),
            # torch.nn.Softmax(dim=1)
        )
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)
        # --------- Lobotomization ---------
        self.is_lobotomized = is_lobotomized

    def get_zeros_initial_state(self, batch_size, overall_or_partial=0):
        zeros_initial_state = torch.zeros(self.n_layers[overall_or_partial], 2, batch_size,
                                          self.hidden_size[overall_or_partial], requires_grad=False, )
        return zeros_initial_state

    def fwd_overall(self, input_tensor, states):
        # --------- Input dense layer ---------
        hx = self.overall_input_dense(input_tensor)  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[0]):
            hx, cx = self.overall_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense[0](hx) + self.logTemperature  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def fwd_partial(self, input_tensor, states):
        # --------- Input dense layer ---------
        hx = self.partial_input_dense(input_tensor)  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[1]):
            hx, cx = self.partial_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense[1](hx) + self.logTemperature  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def forward(self,
                partial_obs,
                overall_obs,
                states_partial,
                states_overall,
                ):
        res_overall, states_overall = self.fwd_overall(overall_obs, states_overall)
        res_partial, states_partial = self.fwd_partial(partial_obs, states_partial)
        res = self.concat_output_layers(torch.cat([res_overall, res_partial, ], dim=1))
        return res, states_overall, states_partial,

    # def forward(self,
    #             input_tensor,  # (batch_size, input_size)
    #             states,  # (n_layers, 2, batch_size, hidden_size)
    #             ):
    #     # --------- Input dense layer ---------
    #     hx = self.input_dense(input_tensor)  # (batch_size, hidden_size)
    #     # layer norm + activation
    #     # --------- Stacked RNN cells ---------
    #     new_states = []  # new states of stacked RNNs
    #     for i in range(self.n_layers):
    #         hx, cx = self.stacked_cells[i](hx,  # (batch_size, hidden_size)
    #                                        (states[i, 0, :, :],  # (batch_size, hidden_size)
    #                                         states[i, 1, :, :]  # (batch_size, hidden_size)
    #                                         ))
    #         new_states.append(torch.stack([hx, cx]))
    #     # --------- Output dense layer ---------
    #     # Probabilities from neural net
    #     res = self.output_dense(hx) + self.logTemperature  # (batch_size, output_size)
    #     # Applying activation function
    #     res = self.output_activation(res)  # (batch_size, output_size)
    #     # Probabilities from random number generator
    #     if self.is_lobotomized:
    #         res = torch.log(torch.rand(res.shape))
    #     out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
    #     # --------------- Return ---------------
    #     return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params

class Embedding_LstmQValueNetwork(torch.nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple,
                 embedding_dim: tuple = (6, 6),
                 n_layers: tuple = (1, 1),
                 stacked_cells=None,
                 output_dense=None,
                 is_lobotomized=False,
                 device="cuda:1",
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        # --------- Embedding layer ---------
        self.overall_embedding = labelEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[0],
                                                input_dim=obs_space_dim[0], out_dim=self.hidden_size[0])
        self.partial_embedding = labelEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[1],
                                                input_dim=3, out_dim=self.hidden_size[1] - (obs_space_dim[1] - 3))
        # --------- Stacked RNN cells ---------
        self.n_layers = n_layers
        overall_stacked_cells, partial_stacked_cells = None, None
        if stacked_cells is None:
            overall_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[0],
                                                                           hidden_size=self.hidden_size[0])
                                                         for _ in range(self.n_layers[0])])
            partial_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[1],
                                                                           hidden_size=self.hidden_size[1])
                                                         for _ in range(self.n_layers[1])])
        self.overall_stacked_cells = overall_stacked_cells
        self.partial_stacked_cells = partial_stacked_cells
        # --------- Output dense layer ---------
        self.output_size = action_space_dim
        if output_dense is None:
            output_dense = []
            output_dense.append(torch.nn.Linear(self.hidden_size[0], self.output_size).to(device))
            output_dense.append(torch.nn.Linear(self.hidden_size[1], self.output_size).to(device))
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x)  # Mapping output to log(p)
        # lambda x: torch.nn.functional.softmax(x, dim=1)
        # torch.sigmoid
        # --------- concat network output ---------
        self.concat_output_layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.output_size, self.output_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.output_size, self.output_size),
            # torch.nn.Softmax(dim=1)
        )
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)
        # --------- Lobotomization ---------
        self.is_lobotomized = is_lobotomized

    def get_zeros_initial_state(self, batch_size, overall_or_partial=0):
        zeros_initial_state = torch.zeros(self.n_layers[overall_or_partial], 2, batch_size,
                                          self.hidden_size[overall_or_partial], requires_grad=False, )
        return zeros_initial_state

    def fwd_overall(self, state_tensor, action_tensor, states):
        # --------- Input dense layer ---------
        hx = self.overall_embedding(state_tensor.int())  # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[0]):
            hx, cx = self.overall_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense[0](hx) + self.logTemperature  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def fwd_partial(self, input_tensor, states):
        # --------- Input dense layer ---------
        hx = self.partial_embedding(input_tensor[:, :3].int())
        hx = torch.cat((hx, input_tensor[:, 3:]), dim=1)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = []  # new states of stacked RNNs
        for i in range(self.n_layers[1]):
            hx, cx = self.partial_stacked_cells[i](hx,  # (batch_size, hidden_size)
                                                   (states[i, 0, :, :],  # (batch_size, hidden_size)
                                                    states[i, 1, :, :]  # (batch_size, hidden_size)
                                                    ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense[1](hx) + self.logTemperature  # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)  # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def forward(self,
                partial_obs,
                overall_obs,
                action,
                states_partial,
                states_overall,
                ):
        res_overall, states_overall = self.fwd_overall(overall_obs, states_overall)
        res_partial, states_partial = self.fwd_partial(partial_obs, states_partial)
        res = self.concat_output_layers(torch.cat([res_overall, res_partial, ], dim=1))
        # res = self.output_activation(res)
        return res, states_overall, states_partial,

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params

class TransformerPolicyNetwork(nn.Module):

    def __init__(self,
                 obs_space_dim,
                 time_embedding_dim,
                 action_space_dim,
                 hidden_dim=256,
                 num_heads=8,
                 num_layers=4):
        super(TransformerPolicyNetwork, self).__init__()

        self.obs_embedding = nn.Linear(obs_space_dim, hidden_dim)

        self.time_embedding = nn.Embedding(time_embedding_dim, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads
        )

        self.num_layers = num_layers

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_dim, action_space_dim)
        self.is_lobotomized = False

    def forward(self, obs, time):
        # Embed input features
        obs_embedded = self.obs_embedding(obs)
        time_embedded = self.time_embedding(time)

        # Combine embeddings
        embedded = obs_embedded + time_embedded

        encoder_output = self.transformer_encoder(embedded)

        decoder_output = self.transformer_decoder(embedded, encoder_output)

        output = self.output_layer(decoder_output)

        # Apply softmax to get action probabilities
        action_probs = F.softmax(output, dim=-1)

        return action_probs

# if __name__ == "__main__":

#     state_space_dim = 64
#     action_space_dim = 10
#     time_embedding_dim = 5
#     batch_size = 10
#     model = TransformerPolicyNetwork(state_space_dim, action_space_dim, time_embedding_dim)

#     state = torch.randn((batch_size, state_space_dim))
#     action = torch.randint(0, 10, (batch_size, 1))
#     time = torch.randint(0, 5, (batch_size, 1))

#     output_probs = model(state, action, time)

#     print("输出概率分布:", output_probs)