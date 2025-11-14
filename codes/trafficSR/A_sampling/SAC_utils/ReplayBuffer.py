import torch
import gc
import time
import collections
import random
import numpy as np

EPS = 1e-3


class ReplayBuffer:
    """基本属性"""

    def __init__(self, capacity=10000, alpha=0.6, rank_based=False):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_experiences = 0
        self.capacity = capacity
        self.alpha = alpha
        self.rank_based = rank_based
        # self.buffer = []

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def pop_old_traj(self):
        while self.n_experiences > self.capacity:
            self.buffer.popleft()
            self.n_experiences -= 1
    
    def update_abs_td_error(self, abs_td_errors):
        for idx in range(len(self.buffer)):
            old_tuple = self.buffer[idx]
            new_tuple = old_tuple[:-1] + (abs_td_errors[idx,0],) #abs_td_errors[idx,0],必须有逗号，不然会报错ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (13,) + inhomogeneous part.
            self.buffer[idx] = new_tuple

    """存储一个采样状态"""

    def add_single_state(
        self,
        obsers_overall,
        obsers_partial,
        actions,
        rewards,
        next_obsers_overall,
        next_obsers_partial,
        dones,
        states_overall=None,
        states_partial=None,
        next_states_overall=None,
        next_states_partial=None,
        prior_UCBs=None,
        next_prior_UCBs=None,
        abs_td_errors=None,
    ):
        # trajectories: (n_add, max_time_step)
        for (
            obser_overall,
            obser_partial,
            action,
            reward,
            next_state_obser_overall,
            next_state_obser_partial,
            done,
            state_overall,
            state_partial,
            next_state_overall,
            next_state_partial,
            prior_UCB,
            next_prior_UCB,
            abs_td_error,
        ) in zip(
            obsers_overall,
            obsers_partial,
            actions,
            rewards,
            next_obsers_overall,
            next_obsers_partial,
            dones,
            states_overall,
            states_partial,
            next_states_overall,
            next_states_partial,
            prior_UCBs,
            next_prior_UCBs,
            abs_td_errors
        ):
            self.buffer.append(
                (
                    obser_overall,
                    obser_partial,
                    action,
                    reward,
                    next_state_obser_overall,
                    next_state_obser_partial,
                    done,
                    state_overall,
                    state_partial,
                    next_state_overall,
                    next_state_partial,
                    prior_UCB,
                    next_prior_UCB,
                    abs_td_error
                )
            )
        self.n_experiences = self.len()
        self.pop_old_traj()

    """存储一条采样序列"""

    def add_trajectory(
        self,
        obsers_overall,
        obsers_partial,
        actions,
        rewards,
        valid_lengths,
        states_overall,
        states_partial,
        prior_UCBs,
    ):
        res_obsers_overall = []
        res_obsers_partial = []
        res_actions = []
        res_rewards = []
        res_next_obsers_overall = []
        res_next_obsers_partial = []
        res_dones = []
        res_states_overall = []
        res_states_partial = []
        res_next_states_overall = []
        res_next_states_partial = []
        res_prior_UCBs = []
        res_next_prior_UCBs = []
        res_abs_td_errors = []
        for idx, (length) in enumerate(valid_lengths):
            for index in range(int(length)):
                res_obsers_overall.append(obsers_overall[idx, index, :])
                res_obsers_partial.append(obsers_partial[idx, index, :])
                res_states_overall.append(states_overall[idx, :, :, index, :])
                res_states_partial.append(states_partial[idx, :, :, index, :])
                res_actions.append(actions[idx, index])
                res_rewards.append(rewards[idx])
                # res_rewards.append(index / length * rewards[idx] if index < int(length) - 1 else rewards[idx])
                # res_rewards.append(0 if index < int(length) - 1 else rewards[idx])
                res_dones.append(False if index < int(length) - 1 else True)
                res_prior_UCBs.append(prior_UCBs[idx, index, :])
                res_abs_td_errors.append(0) # 0 for initial value，之后会由TD error更新
                if index + 1 < obsers_overall.shape[1]:
                    res_next_obsers_overall.append(obsers_overall[idx, index + 1, :])
                    res_next_obsers_partial.append(obsers_partial[idx, index + 1, :])
                    res_next_states_overall.append(
                        states_overall[idx, :, :, index + 1, :]
                    )
                    res_next_states_partial.append(
                        states_partial[idx, :, :, index + 1, :]
                    )
                    res_next_prior_UCBs.append(prior_UCBs[idx, index + 1, :])
                else:
                    res_next_obsers_overall.append(obsers_overall[idx, index, :])
                    res_next_obsers_partial.append(obsers_partial[idx, index, :])
                    res_next_states_overall.append(states_overall[idx, :, :, index, :])
                    res_next_states_partial.append(states_partial[idx, :, :, index, :])
                    res_next_prior_UCBs.append(prior_UCBs[idx, index, :])

        self.add_single_state(
            res_obsers_overall,
            res_obsers_partial,
            res_actions,
            res_rewards,
            res_next_obsers_overall,
            res_next_obsers_partial,
            res_dones,
            res_states_overall,
            res_states_partial,
            res_next_states_overall,
            res_next_states_partial,
            res_prior_UCBs,
            res_next_prior_UCBs,
            res_abs_td_errors,
        )

    """sample相关"""
    
    # 根据排名获得采样概率probs = 1 / (ranks + 1)
    def get_normalized_weights(self):
        # abs_td_errors越大，代表这个样本越难被预测清楚，所以越需要被采样
        probs = np.array(
            [abs_td_error for _, _, _, _, _, _, _, _, _, _, _, _, _, abs_td_error in self.buffer])
        probs = probs + EPS
        # probs = np.array([abs(reward) for _, _, _, reward, _, _, _, _, _, _, _, _, _ in self.buffer]) #旧版本
        # also for - reward；即使没达到done，它中间状态也有reward值，就是最终的reward
        if self.rank_based:  # better？因为如果在rewards相差不大的情况下，其实可以多个类型都取的
            temp = probs.argsort()[::-1]  # large to small
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(probs))
            probs = 1 / (ranks + 1)
        else:
            probs = probs + EPS
        scaled_probs = probs**self.alpha
        pri_sum = np.sum(scaled_probs)
        probs = np.array(scaled_probs / pri_sum, dtype=np.float64)
        return probs

    # 获取轨迹的dict形式
    def transform_transition2dict(self, transitions):
        (
            obsers_overall,
            obsers_partial,
            actions,
            rewards,
            next_obsers_overall,
            next_obsers_partial,
            dones,
            states_overall,
            states_partial,
            next_states_overall,
            next_states_partial,
            prior_UCBs,
            next_prior_UCBs,
            abs_td_errors,
        ) = zip(*transitions)
        # build dataset
        transition_dict = {
            "obsers_overall": obsers_overall,
            "obsers_partial": obsers_partial,
            "actions": actions,
            "rewards": rewards,
            "next_obsers_overall": next_obsers_overall,
            "next_obsers_partial": next_obsers_partial,
            "dones": dones,
            "states_overall": states_overall,
            "states_partial": states_partial,
            "next_states_overall": next_states_overall,
            "next_states_partial": next_states_partial,
            "prior_UCBs": prior_UCBs,
            "next_prior_UCBs": next_prior_UCBs,
            "abs_td_errors": abs_td_errors,
        }
        return transition_dict
    
    # PER：使得那些对学习更有价值的经验被更频繁地采样
    def sample_token_idx_from_buffer(self, batch_size, replace=True):
        probs = self.get_normalized_weights()
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=replace, p=probs)  # replace=True means can repeatedly choose
        transitions = [self.buffer[idx] for idx in indices]
        # transitions = random.choices(self.buffer, weights=probs, k=batch_size) # random.choices默认是可以重复选取的
        transition_dict = self.transform_transition2dict(transitions)
        return transition_dict
