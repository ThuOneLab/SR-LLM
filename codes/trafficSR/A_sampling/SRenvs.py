import torch
import numpy as np
from codes.trafficSR.A_sampling.env_generator.make_single_env import make_env_fn, get_make_env_kargs, \
    divide_number_into_parts
from codes.trafficSR.A_sampling.prior.SRprior import Prior_Collector


class MultiprocessEnv(object):
    def __init__(
            self,
            X_numpy=None, y_numpy=None,
            train_args=None,
            token_args=None,
            ngsim_args=None,
            prefix_tree=None,
            batch_size=2000,
            seed=100,
            n_workers=1,
    ):
        self.batch_size = batch_size
        self.ngsim_args = ngsim_args
        make_env_kargs = get_make_env_kargs(X_numpy=X_numpy, y_numpy=y_numpy,
                                            train_args=train_args,
                                            token_args=token_args,
                                            ngsim_args=ngsim_args,
                                            prefix_tree=prefix_tree, )
        self.prior_args = train_args["prior_args"]
        self.main_env = make_env_fn(**make_env_kargs, seed=seed)
        self.parts = divide_number_into_parts(batch_size, n_workers)
        self.multi_envs = [make_env_fn(**make_env_kargs, seed=seed + rank, batch_size=self.parts[rank]) for rank in
                           range(n_workers)]
        self.n_workers = n_workers
        parts_sum = [0]
        self.parts_sum = parts_sum + [sum(self.parts[:i + 1]) for i in range(len(self.parts))]

    def getTraj(self, train_args, policy_network, prefix_tree, epoch, specific_prog=None):
        observations_overall, observations_partial = [], []
        states_overall, states_partial = [], []
        logits, actions_numpy, R, R_SUB, R_similarity = [], [], [], [], []
        prior_UCBs = []
        for i in range(len(self.multi_envs)):
            self.multi_envs[i]._reset()
            prior_collector = Prior_Collector(self.multi_envs[i], self.prior_args)
            '''get trajectory'''
            (logits_i, actions_numpy_i, observations_overall_i, observations_partial_i, R_i, R_SUB_i,
             R_similarity_i, observations_i, states_overall_i, states_partial_i, prior_UCBs_i) = \
                self.multi_envs[i].getTraj(
                    train_args,
                    policy_network,
                    prior_collector,
                    prefix_tree,
                    epoch,
                    specific_program=specific_prog)
            observations_overall.append(observations_overall_i)
            observations_partial.append(observations_partial_i)
            states_overall.append(states_overall_i)
            states_partial.append(states_partial_i)
            logits.append(logits_i)
            prior_UCBs.append(prior_UCBs_i)
            actions_numpy.append(actions_numpy_i)
            R = np.append(R, R_i)
            R_SUB.append(R_SUB_i)
            R_similarity.append(R_similarity_i)
            self.update_envs_info(env_id=i)
        logits = torch.cat(logits, dim=1)
        actions_numpy = np.concatenate(actions_numpy, axis=1)
        observations_overall = np.concatenate(observations_overall, axis=1)
        observations_partial = np.concatenate(observations_partial, axis=1)
        states_overall = torch.cat(tuple(states_overall), dim=3)
        states_partial = torch.cat(tuple(states_partial), dim=3)
        prior_UCBs = torch.cat(tuple(prior_UCBs), dim=1)
        R_SUB = np.concatenate(R_SUB, axis=0)
        R_similarity = np.concatenate(R_similarity, axis=0)
        return logits, actions_numpy, observations_overall, observations_partial, R, R_SUB, R_similarity, states_overall, states_partial, prior_UCBs

    def get_reward_built_expression(self, prog, layer_of_programs=1):
        self.multi_envs[0]._reset()
        if layer_of_programs == 1:
            for token in prog:
                actions = [self.multi_envs[0].library.all_tokens_id_dict['end']] * self.multi_envs[0].batch_size
                actions[0] = self.multi_envs[0].library.all_tokens_id_dict[token]
                self.multi_envs[0].step(torch.tensor(actions))
            self.multi_envs[0].library.is_physical[0] = True
        elif layer_of_programs == 2:
            max_len = max(len(program) for program in prog)
            # action step
            for token_index in range(max_len):
                actions = [self.multi_envs[0].library.all_tokens_id_dict['end']] * self.multi_envs[0].batch_size
                for i, program in enumerate(prog):
                    if token_index < len(program):
                        token = program[token_index]
                        actions[i] = self.multi_envs[0].library.all_tokens_id_dict[token]
                self.multi_envs[0].step(torch.tensor(actions))
            # set physical
            for i in range(len(prog)):
                self.multi_envs[0].library.is_physical[i] = True

        R, R_SUB, R_similarity = self.multi_envs[0].get_reward(prefix_tree=self.multi_envs[0].prefix_tree)
        return R, R_SUB, R_similarity

    def update_envs_info(self, env_id):
        self.main_env.library.tokens_idx[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].library.tokens_idx
        self.main_env.library.is_physical[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].library.is_physical
        self.main_env.library.have_completed[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].library.have_completed
        self.main_env.library.units_inconsistency[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].library.units_inconsistency
        self.main_env.library.prefix_str[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].library.prefix_str
        self.main_env.programs.valid_lengths[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = self.multi_envs[
            env_id].programs.valid_lengths
        self.main_env.programs.free_const_values.values[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = \
            self.multi_envs[
                env_id].programs.free_const_values.values
        self.main_env.programs.semi_free_const_values.values[self.parts_sum[env_id]:self.parts_sum[env_id + 1]] = \
            self.multi_envs[
                env_id].programs.semi_free_const_values.values

    def false_bool_use_rough_calibration(self):
        for i in range(len(self.multi_envs)):
            self.multi_envs[i].bool_use_rough_calibration = False

    def __getitem__(self, item):
        return self.multi_envs[item]

    def __len__(self):
        return len(self.multi_envs)
