'''
Author: guozelin-ai 3190102461@zju.edu.cn
Date: 2024-10-11 17:21:15
LastEditors: guozelin-ai 3190102461@zju.edu.cn
LastEditTime: 2025-02-18 23:13:51
FilePath: \Symbolic_Regression_with_Large_Language_Models\codes\trafficSR\A_sampling\env_composition\SRprograms.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import copy
import numpy as np
import torch
from codes.trafficSR.A_sampling.env_composition.SRprogram import SRprogram
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import Token, UNITS_VECTOR_SIZE
from codes.trafficSR.A_sampling.env_composition.SRlibrary_utils import FreeConstantsTable
from codes.trafficSR.A_sampling.env_composition.SRlibrary import SRlibrary
from codes.trafficSR.C_evaluation.reward_utils.SRsimilarity import Similarity
import codes.trafficSR.A_sampling.prior.SRdimension_analysis as dimensional_analysis

# from trafficSR.physym.SRreward import get_single_reward, ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY, \
#     DEFAULT_REWARD_FUNCTION, DEFAULT_REWARD_FUNCTION_VELOCITY_TRAJECTORY, DEFAULT_REWARD_FUNCTION_VELOCITY, \
#     DEFAULT_REWARD_FUNCTION_TRAJECTORY, optimize_free_const


INTERFACE_UNITS_UNAVAILABLE = 0.
INTERFACE_UNITS_AVAILABLE = 1.


class AllTokens:
    def __init__(self, all_tokens):
        self.all_tokens = all_tokens
        self._count_all_tokens()

    def _count_all_tokens(
            self,
    ):
        self.variable_tokens_number, \
            self.fixed_const_tokens_number, \
            self.semi_free_const_tokens_number, \
            self.free_const_tokens_number, \
            self.operator_tokens_number, \
            self.combination_tokens_number, \
            self.end_tokens_number = 0, 0, 0, 0, 0, 0, 0
        self.free_const_tokens_position, self.semi_free_const_tokens_position, self.fixed_const_tokens_position = {}, {}, {}

        for tok in self.all_tokens:
            if tok.token_type == 'variable':
                self.variable_tokens_number += 1
            elif tok.token_type == 'fixed_const':
                self.fixed_const_tokens_position[tok.token_name] = self.fixed_const_tokens_number
                self.fixed_const_tokens_number += 1
            elif tok.token_type == 'semi_free_const':
                self.semi_free_const_tokens_position[tok.token_name] = self.semi_free_const_tokens_number
                self.semi_free_const_tokens_number += 1
            elif tok.token_type == 'free_const':
                self.free_const_tokens_position[tok.token_name] = self.free_const_tokens_number
                self.free_const_tokens_number += 1
            elif tok.token_type == 'operator':
                self.operator_tokens_number += 1
            elif tok.token_type == 'combination':
                self.combination_tokens_number += 1
            elif tok.token_type == 'end':
                self.end_tokens_number += 1
            else:
                raise ValueError(f"Unknown token type {tok.token_type}")


def update_library(library:SRlibrary, new_tokens_idx, zero_out_unphysical=False):
    library.update_new_tokens_idx(new_tokens_idx)
    
    library.update_include_free_const()
    library.update_have_end() #看新加的是否是end
    library.update_have_combination() #看新加的是否是combination
    
    library.update_real_length() #发现最终real_length一般都能控制在max_time_step内：因为LengthPrior
    
    # library.update_tokens_one_hot()
    if zero_out_unphysical:
        library.update_is_physical()
        library.update_units_state() # update current_time_step
        
    library.update_n_placeholders()# 更新加入new_token后，上一轮遗留下来几个placeholders(self.n_placeholder_left)和这轮的placeholders数量（self.n_placeholder_now）
    library.update_length_unphysical() # update 没考虑到lengthPrior导致超出self.real_length的unphysical
    
    library.update_have_completed() # update have_completed. 根据n_placeholder_now<=0，判断已经采样完成
    
    library.update_complexity()
    
    library.update_previous_exsiting_placeholders_relations() #更新先前存在的before placeholders的父兄弟子，更新为shift后after placeholders的父兄弟子。另外反过来把（1）after的父节点的子节点index，更新为after；（2）after的兄弟节点的兄弟节点index，更新为after
    library.update_add_placeholders_relations() #更新add placeholder的父兄弟。另外反过来把（1）新加的placeholders的父节点的子节点index，更新为新加的placeholders；（2）新加的placeholders的兄弟节点的兄弟节点index，更新为新加的placeholders
    
    library.current_time_step += 1


class SRprograms:
    def __init__(
            self,
            env_args=None,
            library=None,
            all_tokens=None,
            free_const_values=None,
            semi_free_const_values=None,
            fixed_const_values=None,
            zero_out_unphysical=True,
            n_data_sources=1,
            bool_args=None,
            batch_size=None,
            rng=None,
    ):
        self.batch_size = env_args["batch_size"] if batch_size is None else batch_size
        self.max_time_step = env_args["max_time_step"]
        self.shape = (self.batch_size, self.max_time_step)
        self.zero_out_unphysical = zero_out_unphysical  # Should unphysical programs be zeroed out ?

        self.library = library

        self.free_const_values = FreeConstantsTable(self.batch_size, n_data_sources, free_const_values)
        self.semi_free_const_values = FreeConstantsTable(self.batch_size, n_data_sources, semi_free_const_values)
        self.fixed_const_values = fixed_const_values
        self.bool_use_rough_calibration = bool_args["bool_use_rough_calibration"]

        self.bool_find_new_formula = bool_args["bool_find_new_formula"]

        self.bool_use_true_trajectory = bool_args["bool_use_true_trajectory"]

        self.all_tokens = all_tokens
        self.all_tokens_id_dict = self.library.all_tokens_id_dict
        self.all_tokens_info = AllTokens(self.all_tokens)
        self.token_number = len(self.all_tokens)

        # self.ideal_max_complexity = 1.1 * self.library.super_parent_info.total_complexity if (
        #         self.library.super_parent_info.total_complexity > 0) else 4. / 3. * max_time_step
        self.ideal_max_complexity = 4. / 3. * self.max_time_step
        self.similarity_args = env_args["similarity_args"]
        if self.similarity_args["compute_similarity"]:
            self.get_anchoring_function()
            self.similarity = [
                Similarity(self.library, anchor_prog_id=i, similarity_reward=self.similarity_args["similarity_reward"])
                for i
                in range(len(self.library.super_parent_info.anchor_prog))]
        else:
            self.similarity = None
        self.reward_weight = env_args["reward_weight"]
        self.rng = rng
        self._reset()

    def set_program(self, prefix_expression):
        prog_tokens = []
        for t in range(len(prefix_expression)):
            token_idx = self.all_tokens_id_dict[prefix_expression[t]]
            if token_idx == self.library.all_token_info_table.invalid_id:
                break
            if token_idx in self.library.all_token_info_table.combination_group:
                prog_tokens = prog_tokens + self.all_tokens[token_idx].tokens_list
            else:
                prog_tokens.append(self.all_tokens[token_idx])
        prog = SRprogram(
            all_tokens_info=self.all_tokens_info,
            tokens=prog_tokens,
            free_const_values=self.free_const_values.values[0],
            semi_free_const_values=self.semi_free_const_values.values[0],
            fixed_const_values=self.fixed_const_values,
            subs=self.subs,
        )
        return prog

    def get_anchoring_function(self, ):
        for prog_id in range(len(self.library.super_parent_info.superparent_prog)):
            prog_tokens = []
            for token_name in self.library.super_parent_info.superparent_prog[prog_id]:
                if token_name not in self.library.all_token_info_table.token_name_table:  # string
                    raise ValueError(f"Token {token_name} not in token_name_table")
                prog_tokens.append(self.all_tokens[self.all_tokens_id_dict[token_name]])  # return token_name
            self.library.super_parent_info.anchor_prog.append(SRprogram(
                all_tokens_info=self.all_tokens_info,
                tokens=prog_tokens,
                free_const_values=None,
                fixed_const_values=self.fixed_const_values,
            ))
            self.update_anchor_prog_relation(int(prog_id))

    def update_anchor_prog_relation(self, prog_id):
        library = copy.deepcopy(self.library)
        library._reset()
        for i in range(len(self.library.super_parent_info.anchor_prog[prog_id].tokens)):
            new_tokens_idx = np.full((self.batch_size), fill_value=self.all_tokens_id_dict["end"], dtype=int)
            new_tokens_idx[0] = self.library.super_parent_info.anchor_prog[prog_id].tokens[i].token_id
            update_library(library, new_tokens_idx=new_tokens_idx)
        library.get_height()
        library.get_children_end_and_length()
        self.library.super_parent_info.total_complexity += library.total_complexity[0]
        # self.library.super_parent_info.parent_index[0,:] = copy.deepcopy(library.parent_index[0, :])
        # self.library.super_parent_info.sibling_index[0,:] = copy.deepcopy(library.sibling_index[0, :])
        self.library.super_parent_info.children_index[prog_id, :] = copy.deepcopy(library.children_index[0, :])
        self.library.super_parent_info.children_number[prog_id, :] = copy.deepcopy(library.children_number[0, :])
        self.library.super_parent_info.children_end[prog_id, :] = copy.deepcopy(library.children_end[0, :])
        self.library.super_parent_info.children_length[prog_id, :] = copy.deepcopy(library.children_length[0, :])
        self.library.super_parent_info.height[prog_id, :] = copy.deepcopy(library.height[0, :])

    def init_subs(self):
        self.subs = {}
        for variable_id in self.library.all_token_info_table.variable_group + self.library.all_token_info_table.semi_free_const_group + self.library.all_token_info_table.free_const_group:
            self.subs[self.library.all_token_info_table.token_name_table[variable_id]] = 0.
        for (i, fixed_const_id) in enumerate(self.library.all_token_info_table.fixed_const_group):
            self.subs[self.library.all_token_info_table.token_name_table[fixed_const_id]] = float(
                self.fixed_const_values[i])

    def _reset(
            self
    ):
        self.init_subs()
        self.is_completed = np.zeros(self.shape[1])
        self.completed_length = 0
        self.valid_lengths = np.zeros(self.batch_size)
        self.free_const_values._reset()
        self.semi_free_const_values._reset()
        # self.library._reset()

    def append(
            self,
            new_tokens_idx,
    ):
        # new_idxs: (batch_size, )
        assert new_tokens_idx.shape[0] == self.shape[0], \
            f"new_idxs.shape[0] must be equal to self.shape[0]"
        assert self.is_completed[-1] == 0, \
            f"Cannot append new programs because the last time step is completed"

        this_pos = np.argmax(self.is_completed == 0)
        self.is_completed[this_pos] = 1
        self.library.tokens_idx[:, this_pos] = new_tokens_idx
        self.completed_length += 1
        end_idx = len(self.all_tokens) - 1
        for prog_id in range(self.batch_size):
            if self.library.tokens_idx[prog_id, this_pos] < end_idx:
                self.valid_lengths[prog_id] += 1
            elif self.library.tokens_idx[prog_id, this_pos] == end_idx:
                continue
            else:
                raise ValueError("token_idx out of range!")
        update_library(self.library, new_tokens_idx, self.zero_out_unphysical)

    def __getitem__(
            self,
            prog_idx
    ):

        prog_tokens = []
        for t in range(self.shape[1]):
            if self.library.tokens_idx[prog_idx, t] == self.library.all_token_info_table.invalid_id - 1 or \
                    self.library.tokens_idx[
                        prog_idx, t] == self.library.all_token_info_table.invalid_id:  # meet end or invalid
                break
            if self.library.tokens_idx[prog_idx, t] in self.library.all_token_info_table.combination_group:
                prog_tokens = prog_tokens + self.all_tokens[self.library.tokens_idx[prog_idx, t]].tokens_list
            else:
                prog_tokens.append(self.all_tokens[self.library.tokens_idx[prog_idx, t]])
        prog = SRprogram(
            all_tokens_info=self.all_tokens_info,
            tokens=prog_tokens,
            free_const_values=self.free_const_values.values[prog_idx],
            semi_free_const_values=self.semi_free_const_values.values[prog_idx],
            fixed_const_values=self.fixed_const_values,
            subs=self.subs,
        )
        return prog

    def __repr__(self):
        return str([self[prog_idx].__repr__() for prog_idx in range(self.shape[0])])

    def initialize_unit_obs(self, n_batch):
        # Initialize result with filler (unavailable units everywhere)
        units_obs = np.zeros((n_batch, UNITS_VECTOR_SIZE + 1),
                             dtype=float)  # (batch_size, UNITS_VECTOR_SIZE + 1) 默认设置了最后一位为0
        shape = (n_batch, UNITS_VECTOR_SIZE)
        if self.rng is None:
            units_obs[:, :-1] = np.random.uniform(size=shape, low=-4, high=4)
        else:
            units_obs[:, :-1] = self.rng.uniform(size=shape, low=-4, high=4)  # (batch_size, UNITS_VECTOR_SIZE)
        return units_obs

    def get_one_hot(self, n_batch, parent_idx, mask_have_parent):
        # Initialize one hot result
        one_hot = np.zeros((n_batch, self.token_number))  # (batch_size, n_choices)
        idx_label = np.full(shape=(n_batch), fill_value=self.library.all_token_info_table.invalid_id,
                            dtype=np.float32)
        # Affecting only valid parents and leaving zero vectors where no parents
        one_hot[mask_have_parent, :] = np.eye(self.token_number)[parent_idx[mask_have_parent]]
        idx_label[mask_have_parent] = parent_idx[mask_have_parent]
        return idx_label, one_hot

    def get_parent_info(self, step=None, coords=None):
        if step is None:
            step = self.library.current_time_step
        batch_pos = coords[0, :]  # Position in batch dim   # (n_batch,)
        n_batch = len(batch_pos)
        mask_have_parent, parent_index, parent_idx = self.library.get_parent_idx(time_step=step,
                                                                                 coords=coords)

        def get_parent_units_obs(step=None, coords=None):
            # Initialize result with filler (unavailable units everywhere)
            units_obs = self.initialize_unit_obs(n_batch)  # (batch_size, UNITS_VECTOR_SIZE + 1)
            if step == 0:  # If 0-th step, units are those of superparent
                units_obs[:, :-1] = self.library.super_parent_info.superparent_units  # (batch_size, UNITS_VECTOR_SIZE)
                units_obs[:, -1] = INTERFACE_UNITS_AVAILABLE  # (batch_size,)
                return units_obs

            parent_is_constraining = np.full(shape=n_batch, fill_value=False, dtype=bool)  # (n_batch,)
            parent_phy_units = np.full(shape=(n_batch, 7), fill_value=np.nan,
                                       dtype=float)  # (n_batch,)
            parent_is_constraining[mask_have_parent] = self.library.is_constraining_phy_units[
                batch_pos[mask_have_parent], parent_index[mask_have_parent]]
            parent_phy_units[parent_is_constraining] = self.library.phy_units[
                batch_pos[parent_is_constraining], parent_index[parent_is_constraining]]

            # Putting units of available parents having available units in units_obs
            units_obs[parent_is_constraining, :-1] = parent_phy_units[
                parent_is_constraining]  # (n_is_available, UNITS_VECTOR_SIZE)
            units_obs[parent_is_constraining, -1] = INTERFACE_UNITS_AVAILABLE  # (n_is_available,)
            return units_obs

        idx, one_hot = self.get_one_hot(n_batch, parent_idx, mask_have_parent)
        return idx, one_hot, get_parent_units_obs(step, coords)

    def get_sibling_info(self, step=None, coords=None):
        if step is None:
            step = self.library.current_time_step
        batch_pos = coords[0, :]  # Position in batch dim   # (n_batch,)
        n_batch = len(batch_pos)
        mask_have_sibling, sibling_not_empty, sibling_index, sibling_idx = self.library.get_sibling_idx(
            time_step=step, coords=coords)

        def get_sibling_one_hot():
            # Initialize one hot result
            one_hot = np.zeros((coords.shape[1], self.token_number))  # (batch_size, n_choices)
            # Affecting only valid parents and leaving zero vectors where no parents
            one_hot[sibling_not_empty, :] = np.eye(self.token_number)[sibling_idx[sibling_not_empty]]
            return one_hot

        def get_sibling_units_obs(step=None, coords=None):
            # Initialize result with filler (unavailable units everywhere)
            units_obs = self.initialize_unit_obs(n_batch)  # (batch_size, UNITS_VECTOR_SIZE + 1)
            sibling_is_constraining = np.full(shape=n_batch, fill_value=False, dtype=bool)  # (n_batch,)
            sibling_phy_units = np.full(shape=(n_batch, 7), fill_value=np.nan,
                                        dtype=float)  # (n_batch,)
            sibling_is_constraining[sibling_not_empty] = self.library.is_constraining_phy_units[
                batch_pos[sibling_not_empty], sibling_index[sibling_not_empty]]
            sibling_phy_units[sibling_is_constraining] = self.library.phy_units[
                batch_pos[sibling_is_constraining], sibling_index[sibling_is_constraining]]

            # Putting units of available parents having available units in units_obs
            units_obs[sibling_is_constraining, :-1] = sibling_phy_units[
                sibling_is_constraining]  # (n_is_available, UNITS_VECTOR_SIZE)
            units_obs[sibling_is_constraining, -1] = INTERFACE_UNITS_AVAILABLE  # (n_is_available,)
            return units_obs

        idx, one_hot = self.get_one_hot(n_batch, sibling_idx, sibling_not_empty)
        return idx, one_hot, get_sibling_units_obs(step, coords)

    def get_previous_info(self, step=None, coords=None):
        if step is None:
            step = self.library.current_time_step
        batch_pos = coords[0, :]  # Position in batch dim   # (n_batch,)
        n_batch = len(batch_pos)
        previous_index = coords[1] - 1
        previous_tokens_coords = np.stack((coords[0], previous_index), axis=0)
        previous_tokens_idx = self.library.tokens_idx[tuple(previous_tokens_coords)]
        mask_previous_valid = previous_tokens_idx < self.token_number - 1  # not end or invalid

        def get_previous_one_hot():
            one_hot = np.zeros((n_batch, self.token_number))  # (batch_size, n_choices)
            if step > 0:
                one_hot[mask_previous_valid, :] = np.eye(self.token_number)[previous_tokens_idx[mask_previous_valid]]
            return one_hot

        def get_previous_units_obs(step=None, coords=None):
            # Initialize result with filler (unavailable units everywhere)
            units_obs = self.initialize_unit_obs(n_batch)  # (batch_size, UNITS_VECTOR_SIZE + 1)
            previous_is_constraining = np.full(shape=n_batch, fill_value=False, dtype=bool)  # (n_batch,)
            previous_phy_units = np.full(shape=(n_batch, 7), fill_value=np.nan,
                                         dtype=float)  # (n_batch,)
            previous_is_constraining[mask_previous_valid] = self.library.is_constraining_phy_units[
                batch_pos[mask_previous_valid], previous_index[mask_previous_valid]]
            previous_phy_units[previous_is_constraining] = self.library.phy_units[
                batch_pos[previous_is_constraining], previous_index[previous_is_constraining]]

            # Putting units of available parents having available units in units_obs
            units_obs[previous_is_constraining, :-1] = previous_phy_units[
                previous_is_constraining]  # (n_is_available, UNITS_VECTOR_SIZE)
            units_obs[previous_is_constraining, -1] = INTERFACE_UNITS_AVAILABLE  # (n_is_available,)
            return units_obs

        idx, one_hot = self.get_one_hot(n_batch, previous_tokens_idx, mask_previous_valid)
        return idx, one_hot, get_previous_units_obs(step, coords)

    def get_current_info(self, step=None, coords=None):
        if step is None:
            step = self.library.current_time_step
        batch_pos = coords[0, :]  # Position in batch dim   # (n_batch,)
        n_batch = len(batch_pos)

        def get_current_units_obs(step=None, coords=None):
            # Initialize result with filler (unavailable units everywhere)
            units_obs = self.initialize_unit_obs(n_batch)  # (batch_size, UNITS_VECTOR_SIZE + 1)
            current_phy_units = np.full(shape=(n_batch, 7), fill_value=np.nan,
                                        dtype=float)  # (n_batch,)
            current_is_constraining = self.library.is_constraining_phy_units[batch_pos, coords[1]]
            current_phy_units[current_is_constraining] = self.library.phy_units[
                batch_pos[current_is_constraining], coords[1][current_is_constraining]]

            # Putting units of available parents having available units in units_obs
            units_obs[current_is_constraining, :-1] = current_phy_units[
                current_is_constraining]  # (n_is_available, UNITS_VECTOR_SIZE)
            units_obs[current_is_constraining, -1] = INTERFACE_UNITS_AVAILABLE  # (n_is_available,)
            return units_obs

        return get_current_units_obs(step, coords)

    '''
    def get_single_reward(
            self,
            prog_id,
            X_numpy,
            y_numpy,
            ngsim_args=None,
            reward_function: list = [ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY],
            similarity=None,
            similarity_compute_limit=0.,
            reward_weight: dict = None,
            device="cpu",
    ):
        X_tensor = torch.tensor(X_numpy).to(device,
                                            dtype=torch.float32)
        y_tensor = torch.tensor(y_numpy).to(device, dtype=torch.float32)
        if reward_function is None:
            reward_function = [DEFAULT_REWARD_FUNCTION]
        prog = self[prog_id]
        if self.library.is_physical[prog_id]:
            if prog.include_free_const:
                lb = self.library.all_token_info_table.const_lb
                ub = self.library.all_token_info_table.const_ub
                for i in range(ngsim_args["n_data_sources"]):
                    start_pos = ngsim_args["stop_positions"][i - 1] if i > 0 else 0
                    end_pos = ngsim_args["stop_positions"][i]
                    init_values = {"init_free_const_values": self.free_const_values.values[prog_id][i],
                                   "init_semi_free_const_values": self.semi_free_const_values.values[prog_id][i],
                                   "bool_use_rough_calibration": self.bool_use_rough_calibration,
                                   "now_data_source_id": i,
                                   "bounds": list(zip(lb, ub)),
                                   }
                    free_const, semi_free_const = optimize_free_const(prog, init_values,
                                                                      X_tensor[start_pos:end_pos + 1],
                                                                      y_tensor[
                                                                      start_pos:end_pos + 1],
                                                                      ngsim_args=ngsim_args)
                    self.free_const_values.values[prog_id][i] = free_const.detach()
                    self.semi_free_const_values.values[prog_id][i] = semi_free_const.detach()
                    if torch.isnan(self.free_const_values.values[prog_id][i]).any().item() or torch.isnan(
                            self.semi_free_const_values.values[prog_id][i]).any().item():
                        self.free_const_values.values[prog_id][i] = torch.tensor(self.free_const_values.init_val)
                        return 0., np.zeros(3 + ngsim_args["n_data_sources"] * 2)
        else:  # unphisical
            return 0., np.zeros(3 + ngsim_args["n_data_sources"] * 2)

        y_pred = prog(X_tensor)
        rmse_rewards = []
        xrmse_reward, vrmse_reward = 0., 0.
        xrmse_rewards, vrmse_rewards = [], []
        for i in range(len(reward_function)):
            if reward_function[i] == DEFAULT_REWARD_FUNCTION:
                rmse_rewards.append(float(reward_function[i](y_pred, y_tensor)))
            elif reward_function[i] == DEFAULT_REWARD_FUNCTION_VELOCITY_TRAJECTORY:
                xrmse_reward, vrmse_reward = reward_function[i](prog, ngsim_args["x_lead"], ngsim_args["v_lead"],
                                                                ngsim_args["x_obs"], ngsim_args["v_obs"],
                                                                ngsim_args["dt"])
                xrmse_reward = float(xrmse_reward)
                vrmse_reward = float(vrmse_reward)
                if math.isnan(xrmse_reward) or math.isnan(vrmse_reward):
                    pass
                else:
                    rmse_rewards.append(xrmse_reward)
                    rmse_rewards.append(vrmse_reward)
            elif reward_function[i] == ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY:
                xrmse_rewards, vrmse_rewards, _, _, _ = reward_function[i](prog, ngsim_args,
                                                                           self.free_const_values.values[prog_id],
                                                                           self.semi_free_const_values.values[prog_id])
                rmse_rewards = rmse_rewards + xrmse_rewards
                rmse_rewards = rmse_rewards + vrmse_rewards
            elif reward_function[i] == DEFAULT_REWARD_FUNCTION_VELOCITY:
                vrmse_reward = float(
                    reward_function[i](y_pred, ngsim_args["x_0"], ngsim_args["v_0"], ngsim_args["v_obs"],
                                       ngsim_args["dt"]))
                rmse_rewards.append(vrmse_reward)
            elif reward_function[i] == DEFAULT_REWARD_FUNCTION_TRAJECTORY:
                xrmse_reward = float(
                    reward_function[i](y_pred, ngsim_args["x_0"], ngsim_args["v_0"], ngsim_args["x_obs"],
                                       ngsim_args["dt"]))
                rmse_rewards.append(xrmse_reward)

        rmse_reward = sum(rmse_rewards) / len(rmse_rewards) if len(rmse_rewards) > 0 else 0.
        factor_complexity = reward_weight['factor_complexity'] if reward_weight is not None else 0.1
        factor_similarity = reward_weight['factor_similarity'] if reward_weight is not None else 0.2
        factor_rmse = reward_weight['factor_rmse'] if reward_weight is not None else 1
        sum_reward = factor_rmse * rmse_reward
        similarity_reward = 0.
        # complexity_reward = torch.exp(torch.tensor(
        #     -abs(progs.library.total_complexity[prog_id] - progs.ideal_max_complexity) / progs.ideal_max_complexity))
        complexity_reward = math.exp(
            -self.library.total_complexity[prog_id] / self.ideal_max_complexity)
        sum_reward += factor_complexity * complexity_reward
        if similarity is not None and rmse_reward > similarity_compute_limit:
            for i in range(len(similarity)):
                similarity_reward += similarity[i].compute(prog, prog_id)
            similarity_reward = similarity_reward / len(similarity)
            sum_reward += factor_similarity * similarity_reward
        if sum_reward >= 1.15 and similarity_reward >= 0.8:
            print("origin/complexity/similarity/sum reward: ", rmse_reward, complexity_reward,
                  similarity_reward, sum_reward)
            print("prog_id and prog: ", prog_id, prog)
        # return sum_reward, np.array([complexity_reward, similarity_reward, rmse_reward, xrmse_reward, vrmse_reward])
        return sum_reward, np.array([complexity_reward, similarity_reward, rmse_reward] + xrmse_rewards + vrmse_rewards)
    
    def get_reward(
            self,
            X_numpy,
            y_numpy,
            prefix_tree,
            ngsim_args=None,
            parrallel_mode=False,
            n_cpus=os.cpu_count(),
            device="cpu",
    ):
        self.library.get_height_and_children_end()  # for similarity reward
        rewards = []
        sub_rewards = []
        all_rewards = []

        # 1.optimization all parameters

        if parrallel_mode:
            pool = mp.Pool(processes=n_cpus)
            for prog_id in range(self.shape[0]):
                # reward, sub_reward = pool.apply_async(get_single_reward, args=(
                #     self, prog_id, X, y, None, self.similarity, self.reward_weight))
                all_reward = pool.apply_async(get_single_reward, args=(
                    self, prog_id, X_numpy, y_numpy, ngsim_args,
                    [ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY], self.similarity,
                    self.similarity_args[
                        "similarity_compute_limit"], self.reward_weight, device))
                all_rewards.append(all_reward)
                # rewards.append(all_reward)
                # sub_rewards.append(sub_reward)
                # if prog_id % 100 == 0:
                #     print(prog_id)

                all_rewards = [all_reward.get() for all_reward in all_rewards]
                rewards = [all_reward[0] for all_reward in all_rewards]
                sub_rewards = [all_reward[1] for all_reward in all_rewards]
                # rewards_new = []
                # idx = 0
                # for reward in rewards:
                #     rewards_new.append(reward.get())
                #     if idx % 100 == 0:
                #         print(idx)
                #     idx+=1
                pool.close()
                pool.join()

        else:
            for prog_id in range(self.shape[0]):
                # tokens_idx = list(self.library.tokens_idx[prog_id])
                # prefix_sequence = [prefix_tree.token_names_lib[token_idx] for token_idx in tokens_idx if
                #                    prefix_tree.token_names_lib[token_idx] != "end"]
                # key = prefix_tree.prefix_sequence_to_key(prefix_sequence)
                key = self.library.prefix_str[prog_id]
                if key in prefix_tree.db.keys():
                    reward_dict = prefix_tree.db[key].reward_dict
                    rewards.append(reward_dict['sum_reward'])
                    sub_rewards.append(np.array(
                        [reward_dict['complexity']['reward'], reward_dict['similarity']['reward'],
                         reward_dict['rmse']['reward'],
                         reward_dict['xrmse']['reward'],
                         reward_dict['vrmse'][
                             'reward']]))  # need to add xrmse and vrmse with different trajectories: have write new get_reward
                    continue
                reward, sub_reward = get_single_reward(self, prog_id, X_numpy, y_numpy, ngsim_args=ngsim_args,
                                                       reward_function=[ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY],
                                                       similarity=self.similarity,
                                                       similarity_compute_limit=self.similarity_args[
                                                           "similarity_compute_limit"],
                                                       reward_weight=self.reward_weight,
                                                       device=device)

                rewards.append(reward)
                sub_rewards.append(sub_reward)

        self.rewards = rewards
        self.sub_rewards = sub_rewards

        return np.array(rewards), np.array(sub_rewards)
    '''

    def get_reward_new(
            self,
            X_numpy,
            y_numpy,
            prefix_tree,
            ngsim_args=None,
            parrallel_mode=False,
            n_cpus=os.cpu_count(),
            device="cpu",
    ):
        # local import
        from codes.trafficSR.B_calibration.SRoptimize_free_const import BatchFreeConstOpti
        from codes.trafficSR.C_evaluation.SRcompute_reward import BatchCalReward
        from codes.trafficSR.C_evaluation.reward_utils.SRreward_function import ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY, \
            DEFAULT_REWARD_FUNCTION
        X_tensor = torch.tensor(X_numpy).to(device,
                                            dtype=torch.float32)
        y_tensor = torch.tensor(y_numpy).to(device, dtype=torch.float32)
        self.library.get_height()  # for similarity reward
        self.library.get_children_end_and_length()  # for similarity reward
        sum_rewards = np.zeros(self.shape[0])
        sub_rewards = np.zeros((self.shape[0], 3 + ngsim_args["n_data_sources"] * 2))
        similarity_rewards = np.zeros(
            (self.shape[0], len(self.similarity))) if self.similarity is not None else None

        # 0.prefix_tree get calculated reward
        # mask_need_calculate = copy.deepcopy(self.library.is_physical) # mask: should calculate reward，不满足物理约束的就不用计算reward
        mask_need_calculate = copy.deepcopy(self.library.is_physical & self.library.have_completed)  # mask: should calculate reward，不满足物理约束的就不用计算reward
        prog_ids = np.where(mask_need_calculate)[0]
        for prog_id in prog_ids:
            key = self.library.prefix_str[prog_id]
            if key in prefix_tree.db.keys():
                mask_need_calculate[prog_id] = False
                reward_dict = prefix_tree.db[key].reward_dict
                sum_rewards[prog_id] = reward_dict['sum_reward']
                sub_rewards[prog_id] = reward_dict['sub_rewards']
                self.free_const_values.values[prog_id] = prefix_tree.db[key].free_const_values
                self.semi_free_const_values.values[prog_id] = prefix_tree.db[key].semi_free_const_values
        # 1.optimization all physical and not calculated parameters
        prog_ids = np.where(mask_need_calculate)[0]
        if len(prog_ids) > 0:
            mask_opti_success = BatchFreeConstOpti(self, prog_ids, X_tensor, y_tensor, ngsim_args=ngsim_args,
                                                   parrallel_mode=parrallel_mode, n_cpus=n_cpus)
            mask_need_calculate[prog_ids] = mask_opti_success
        # 2.calculate reward of useful expressions
        prog_ids = np.where(mask_need_calculate)[0]
        if len(prog_ids) > 0:
            sum_rewards[prog_ids], sub_rewards[prog_ids], similarity_rewards[prog_ids] = BatchCalReward(self, prog_ids,
                                                                                                        X_tensor,
                                                                                                        y_tensor,
                                                                                                        ngsim_args=ngsim_args,
                                                                                                        reward_function=[
                                                                                                            ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY] if
                                                                                                        self.bool_use_true_trajectory
                                                                                                        else [
                                                                                                            DEFAULT_REWARD_FUNCTION],
                                                                                                        # DEFAULT_REWARD_FUNCTION for false data
                                                                                                        similarity=self.similarity,
                                                                                                        similarity_compute_limit=
                                                                                                        self.similarity_args[
                                                                                                            "similarity_compute_limit"],
                                                                                                        reward_weight=self.reward_weight,
                                                                                                        parrallel_mode=parrallel_mode,
                                                                                                        n_cpus=n_cpus)
        # 3.get value and return
        self.rewards = sum_rewards
        self.sub_rewards = sub_rewards
        self.similarity_rewards = similarity_rewards
        return sum_rewards, sub_rewards, similarity_rewards

    def progs_assign_required_units(self):
        # mask : should assign_required_units be run on program ?
        # Do run only on incomplete programs AND physical(?)
        mask_do_run = (~self.library.have_completed) & (self.library.is_physical)
        # mask_do_run = (~self.library.have_completed)
        batch_id = np.arange(start=0, stop=self.batch_size, step=1)
        coords = np.stack(
            (batch_id[mask_do_run], np.full(shape=sum(mask_do_run), fill_value=self.library.current_time_step)), axis=0)
        # print("n_progs_do_assign_required_units", sum(mask_do_run))
        if coords.shape[1] != 0:  # not empty
            dimensional_analysis.assign_required_units(self, coords)


if __name__ == '__main__':
    program = SRprogram(
        tokens=[
            Token(
                name='add',
                type='operator',
                id=0,
                func=np.add,
            ),
            Token(
                name='x',
                type='variable',
                id=0,
            ),
            Token(
                name='y',
                type='variable',
                id=1,
            ),
        ]
    )
    y = program.execute(np.array([1, 2]))
