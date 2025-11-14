import torch
import numpy as np
from codes.trafficSR.A_sampling.policy_network.SRmodel_utils import *
from codes.trafficSR.A_sampling.policy_network.SRmodel_old import *

def get_model(
        model_type="lstm",
        **kwargs,
):
    if model_type == "overall_lstm":
        return SingleLstmPolicyNetwork(**kwargs)
    elif model_type == "lstm":
        return LstmPolicyNetwork(**kwargs)
    elif model_type == "embedding_lstm":
        return Embedding_LstmPolicyNetwork(**kwargs)
    elif model_type == "combined_lstm":
        return CombinedLstmPolicyNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

class Embedding_LstmPolicyNetwork(torch.nn.Module):
    def __init__(self,
                 obs_space_dim: tuple,
                 action_space_dim,
                 hidden_size: tuple,
                 embedding_dim: tuple = (6, 6),
                 n_layers: tuple = (1, 1),
                 overall_and_partial: tuple = (True, True),
                 stacked_cells=None,
                 output_dense=None,
                 is_lobotomized=False,
                 device="cuda:1",
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_overall, self.use_partial = overall_and_partial
        # --------- Embedding layer ---------
        self.overall_embedding = labelEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[0], input_dim=obs_space_dim[0], out_dim=self.hidden_size[0])
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
        dim = 2 if (self.use_partial and self.use_overall) else 1
        self.concat_output_layers = torch.nn.Linear(dim * self.output_size,
                                                    self.output_size)
        # self.concat_output_layers = torch.nn.Sequential(
        #     torch.nn.Linear((self.use_overall + self.use_partial) * self.output_size, self.output_size),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear(self.output_size, self.output_size),
        #     # torch.nn.Softmax(dim=1)
        # )
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
        hx = self.overall_embedding(input_tensor.int())  # (batch_size, hidden_size)
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
        res = torch.add(self.output_dense[1](hx), self.logTemperature)  # (batch_size, output_size)
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
        # res = self.output_activation(res)
        return res, states_overall, states_partial

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params
class CombinedLstmPolicyNetwork(torch.nn.Module):
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
        
        # --------- Overall Embedding layer ---------
        self.overall_token_idx_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[0], input_dim=obs_space_dim[0], out_dim=self.hidden_size[0])
        '''
        # --------- Partial Embedding layer ---------
        self.partial_token_idx_embedding = newEmbedding(action_space_dim=action_space_dim + 1, embedding_dim=embedding_dim[1], input_dim=1, out_dim=self.hidden_size[1]) #  - (obs_space_dim[1] - 3)
        self.partial_placeholder_embedding = newEmbedding(action_space_dim=obs_space_dim[0]*2, embedding_dim=embedding_dim[1], input_dim=1, out_dim=self.hidden_size[1]) #action_space_dim最多有max_time_step*2个placeholder
        self.partial_units_control_embedding = newEmbedding(action_space_dim=2, embedding_dim=embedding_dim[1], input_dim=1, out_dim=self.hidden_size[1]) # 0代表不控制，1代表控制
        # --------- Partial Dense layer ---------
        self.partial_units7_dense = torch.nn.Linear(int((obs_space_dim[1] - 3 -1 -3)/3), self.hidden_size[1]) #去掉了父兄前的token_idx, n_placeholder, units_control
        self.partial_units_dense = torch.nn.Linear(int((obs_space_dim[1] - 3 -1)), self.hidden_size[1]) #去掉了父兄前的token_idx, n_placeholder
        # --------- Partial Output layer ---------
        self.partial_output_dense = torch.nn.Linear(self.hidden_size[1]*5, self.hidden_size[1])
        '''
        
        # --------- Partial Onehot Dense layer ---------
        self.partial_onehot_dense = torch.nn.Linear(obs_space_dim[1], self.hidden_size[1])
        
        # --------- Stacked RNN cells ---------
        self.n_layers = n_layers
        overall_stacked_cells, partial_stacked_cells = None, None
        if stacked_cells is None:
            overall_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[0],hidden_size=self.hidden_size[0]) for _ in range(self.n_layers[0])])
            partial_stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size[1],hidden_size=self.hidden_size[1]) for _ in range(self.n_layers[1])])
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
        hx = self.overall_token_idx_embedding(input_tensor.int())  # (batch_size, hidden_size)
        # hx = self.overall_input_dense(input_tensor)  # (batch_size, hidden_size)
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
        hx1 = self.partial_token_idx_embedding(input_tensor[:, :1].int())
        hx2 = self.partial_token_idx_embedding(input_tensor[:, 1:2].int())
        hx3 = self.partial_token_idx_embedding(input_tensor[:, 2:3].int())
        hx4 = self.partial_placeholder_embedding(input_tensor[:, 3:4].int()) # 最小都为0，不会出现负数
        hx5 = self.partial_units_dense(input_tensor[:, 4:]) # units all
        hx = torch.cat((hx1, hx2, hx3, hx4, hx5), dim=1)
        hx= self.partial_output_dense(F.relu(hx))
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
    
    def fwd_partial_onehot(self, input_tensor, states):
        # --------- Input dense layer ---------
        hx = self.partial_onehot_dense(input_tensor)
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
            # res_overall_cloned = res_overall.clone()
        if self.use_partial:
            res_partial, states_partial = self.fwd_partial_onehot(partial_obs, states_partial)
            # res_partial_cloned = res_partial.clone() #res_partial.clone() 只会克隆张量的值，而不会克隆其梯度。如果你想克隆张量及其梯度，可以使用 torch.autograd.Variable 的 clone 方法，并设置 requires_grad=True。res_partial_cloned = res_partial.clone().detach().requires_grad_(res_partial.requires_grad)

        if self.use_overall and self.use_partial:
            res_sum = torch.cat([res_overall, res_partial, ], dim=1)
            res = self.concat_output_layers(res_sum)
            res=self.output_activation(res)
        elif self.use_overall:
            # res = self.concat_output_layers(res_overall)
            res=res_overall
        elif self.use_partial:
            # res = self.concat_output_layers(res_partial)
            res=res_partial
        else:
            raise ValueError("No input to the network")
        return res, states_overall, states_partial

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params

# class PartialOnehotLstm(torch.nn.Module):
    
if __name__ == "__main__":
    model_args = {
        'obs_space_dim': 64,
        'time_embedding_dim': 5,
        'action_space_dim': 10,
    }
    model = get_model(
        model_type="transformer",
        **model_args,
    )
    batch_size = 10
    state_space_dim = model_args['obs_space_dim']
    state = torch.randn((batch_size, state_space_dim))
    time = torch.randint(0, 5, (batch_size, 1))

    output_probs = model(state, time)

    print(output_probs.shape)
