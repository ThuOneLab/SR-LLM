import os
import math
import datetime

# print("SRCalReward_start_import", datetime.datetime.now())
from multiprocessing import Pool
import numpy as np
from codes.trafficSR.C_evaluation.reward_utils.SRreward_function import ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY, \
    DEFAULT_REWARD_FUNCTION

# print("SRCalReward_end_import", datetime.datetime.now())


def SingleCalReward(prog_id, similarity, progs, X_tensor, y_tensor, ngsim_args=None, reward_function=None,
                    reward_weight=None, ):
    # print("Process Id", os.getpid())
    # print("start", prog_id, datetime.datetime.now())
    rmse_rewards = []
    xrmse_rewards, vrmse_rewards = [], []
    prog = progs[prog_id]
    for i in range(len(reward_function)):
        if reward_function[i] == ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY:
            xrmse_rewards, vrmse_rewards, _, _, _ = reward_function[i](prog, ngsim_args,
                                                                       progs.free_const_values.values[prog_id],
                                                                       progs.semi_free_const_values.values[prog_id])
            rmse_rewards = rmse_rewards + xrmse_rewards
            rmse_rewards = rmse_rewards + vrmse_rewards
        elif reward_function[i] == DEFAULT_REWARD_FUNCTION:
            y_pred = prog(X_tensor)
            rmse_rewards.append(float(reward_function[i](y_pred, y_tensor)))

    rmse_reward = sum(rmse_rewards) / len(rmse_rewards) if len(rmse_rewards) > 0 else 0.
    # for example: (['div', 'delta_v', 'div', 'sub', 's', 'div', 'v', 'div', 'v', 's', 'v']) = delta_v*v/(s-s),so all number=inf or -inf, so the rmse_reward=nan
    if math.isnan(rmse_reward):
        rmse_reward = 0.
    factor_complexity = reward_weight['factor_complexity'] if reward_weight is not None else 0.025
    factor_similarity = reward_weight['factor_similarity'] if reward_weight is not None else 0.1
    factor_rmse = reward_weight['factor_rmse'] if reward_weight is not None else 0.875
    sum_reward = factor_rmse * rmse_reward

    complexity_reward = math.exp(
        -progs.library.total_complexity[prog_id] / progs.ideal_max_complexity)
    sum_reward = sum_reward + factor_complexity * complexity_reward
    similarity_reward = 0.
    similarity_scores = []
    # if similarity is not None and rmse_reward > similarity_compute_limit:
    if similarity is not None:
        for i in range(len(similarity)):
            similarity_score = similarity[i].compute(prog, prog_id, first_input=True if i == 0 else False)
            similarity_scores.append(similarity_score)
            similarity_reward = max(similarity_reward, similarity_score)
        # similarity_reward = similarity_reward / len(similarity)
        sum_reward = sum_reward + factor_similarity * similarity_reward
    # print("end", prog_id, datetime.datetime.now())
    if xrmse_rewards == []:
        xrmse_rewards = [0]
    if vrmse_rewards == []:
        vrmse_rewards = [0]
    return sum_reward, [complexity_reward, similarity_reward,
                        rmse_reward] + xrmse_rewards + vrmse_rewards, similarity_scores


def PartitionCalReward(sub_array, similarity, progs, X_tensor, y_tensor, ngsim_args=None, reward_function=None,
                       reward_weight=None, ):
    # print("Process Id", os.getpid())
    # print("start", sub_array, datetime.datetime.now())
    sum_rewards_results = []
    sub_rewards_results = []
    similarity_rewards_results = []
    for i, prog_id in enumerate(sub_array):
        sum_reward, sub_reward, similarity_scores = SingleCalReward(prog_id, similarity, progs, X_tensor, y_tensor,
                                                                    ngsim_args=ngsim_args,
                                                                    reward_function=reward_function,
                                                                    reward_weight=reward_weight)
        sum_rewards_results.append(sum_reward)
        sub_rewards_results.append(sub_reward)
        similarity_rewards_results.append(similarity_scores)
    # print("end", sub_array, datetime.datetime.now())
    return sum_rewards_results, sub_rewards_results, similarity_rewards_results


def BatchCalReward(progs, prog_ids, X_tensor, y_tensor, ngsim_args=None, reward_function=None, similarity=None,
                   similarity_compute_limit=0., reward_weight=None, parrallel_mode=False, n_cpus=os.cpu_count()):
    sum_rewards = np.zeros(len(prog_ids))
    if not parrallel_mode or prog_ids.shape[0] < 80:
        sub_rewards = np.zeros((len(prog_ids), 3 + ngsim_args["n_data_sources"] * 2))
        similarity_rewards = np.zeros((len(prog_ids), len(similarity)))
        for i, prog_id in enumerate(prog_ids):
            sum_reward, sub_reward, similarity_reward = SingleCalReward(prog_id, similarity, progs, X_tensor, y_tensor,
                                                                        ngsim_args=ngsim_args,
                                                                        reward_function=reward_function,
                                                                        reward_weight=reward_weight)
            sum_rewards[i], sub_rewards[i], similarity_rewards[i] = np.array(sum_reward), np.array(
                sub_reward), np.array(similarity_reward)

    else:
        pool = Pool(processes=n_cpus)
        results = []
        sub_arrays = np.array_split(prog_ids, int(prog_ids.shape[0] / 100) + 1)
        # print("start1", datetime.datetime.now())
        for i, sub_array in enumerate(sub_arrays):
            result = pool.apply_async(PartitionCalReward, args=(
                sub_array, similarity, progs, X_tensor, y_tensor, ngsim_args,
                reward_function,
                reward_weight))
            results.append(result)
        # print("start2", datetime.datetime.now())
        pool.close()
        print("start3", datetime.datetime.now())
        pool.join()
        # print("start4", datetime.datetime.now())
        results = [result.get() for result in results]
        # print("start5", datetime.datetime.now())
        sum_rewards = np.array([sum_reward for result_item in results for sum_reward in result_item[0]])
        sub_rewards = np.array([sub_reward for result_item in results for sub_reward in result_item[1]])
        similarity_rewards = np.array(
            [similarity_reward for result_item in results for similarity_reward in result_item[2]])
    return sum_rewards, sub_rewards, similarity_rewards
