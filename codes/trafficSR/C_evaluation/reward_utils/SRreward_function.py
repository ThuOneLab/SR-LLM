# import datetime
# print("SRreward_function_start_import", datetime.datetime.now())
from torch import zeros, zeros_like, tensor, std, mean, sqrt, isnan


# print("SRreward_function_end_import", datetime.datetime.now())

def use_acc_simulate_v(v0, acc, dt, v2_obs=None):
    delta_v = acc * dt
    v2_sim = zeros(acc.shape[0])
    v2_sim[0] = v0
    for i in range(1, acc.shape[0]):  # don't use last one, because it is out of v/x range
        v2_sim[i] = v2_obs[i - 1] + delta_v[i - 1]
    return v2_sim


def GLOBAL_SINGLE_TRAJECTORY(prog, now_data_source_id=0, ngsim_args=None, free_const_value_i=None,
                             semi_free_const_value_i=None, need_reward=False):
    i = now_data_source_id
    start_pos = ngsim_args["stop_positions"][
                    i - 1] + 2 if i > 0 else 1  # +2 because end(i-1),start,start+1. start+1 begin simulation
    end_pos = ngsim_args["stop_positions"][i]
    preceeding_length = ngsim_args["preceeding_length"][i]
    vector_length = end_pos - start_pos + 1 + 1
    acc2_single = zeros(vector_length)
    v2_single = zeros(vector_length)
    x2_single = zeros(vector_length)

    v2_single[0] = ngsim_args["v_obs"][start_pos - 1]  # initialize start
    x2_single[0] = ngsim_args["x_obs"][start_pos - 1]

    for k in range(1, vector_length):
        X = tensor([x2_single[k - 1] - ngsim_args["x_lead"][k - 1] - preceeding_length, v2_single[k - 1],
                    v2_single[k - 1] - ngsim_args["v_lead"][k - 1]]).reshape(1, -1)
        acc2_single[k - 1] = prog.execute(X, free_const_values=free_const_value_i,
                                          semi_free_const_values=semi_free_const_value_i)
        # acc2_single[k] = clip(acc2_single[k], -50, 50)
        v2_single[k] = v2_single[k - 1] + acc2_single[k - 1] * ngsim_args["dt"]
        if v2_single[k] < 0:
            v2_single[k] = 0
            acc2_single[k - 1] = (v2_single[k] - v2_single[k - 1]) / ngsim_args["dt"]
        x2_single[k] = x2_single[k - 1] + v2_single[k] * ngsim_args["dt"]
    reward_v = 0
    reward_x = 0
    if need_reward:
        std_v = std(ngsim_args["v_obs"][start_pos + 1:end_pos + 1])
        std_x = std(ngsim_args["x_obs"][start_pos + 1:end_pos + 1])
        reward_v = 1 / (1 + 1 / std_v * sqrt(
            mean(
                (v2_single - ngsim_args["v_obs"][start_pos - 1:end_pos + 1]) ** 2)))
        reward_x = 1 / (1 + 1 / std_x * sqrt(
            mean(
                (x2_single - ngsim_args["x_obs"][start_pos - 1:end_pos + 1]) ** 2)))
        reward_v = 0 if isnan(reward_v).any().item() else reward_v
        reward_x = 0 if isnan(reward_x).any().item() else reward_x
    return reward_v, reward_x, acc2_single, v2_single, x2_single


def ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY(prog, ngsim_args=None, free_const_values=None,
                                            semi_free_const_values=None, ):
    acc2_new = zeros(ngsim_args["x_lead"].shape[0])
    v2_new = zeros(ngsim_args["x_lead"].shape[0])
    x2_new = zeros(ngsim_args["x_lead"].shape[0])
    rewards_v, rewards_x = [], []
    # (x, v, delta_v)
    for i in range(ngsim_args["n_data_sources"]):
        free_const_value_i = free_const_values[i]
        semi_free_const_value_i = semi_free_const_values[i]
        reward_v, reward_x, acc2_single, v2_single, x2_single = GLOBAL_SINGLE_TRAJECTORY(prog, now_data_source_id=i,
                                                                                         ngsim_args=ngsim_args,
                                                                                         free_const_value_i=free_const_value_i,
                                                                                         semi_free_const_value_i=semi_free_const_value_i,
                                                                                         need_reward=True)
        start_pos = ngsim_args["stop_positions"][i - 1] + 2 if i > 0 else 1
        end_pos = ngsim_args["stop_positions"][i]
        acc2_new[start_pos - 1:end_pos + 1] = acc2_single
        v2_new[start_pos - 1:end_pos + 1] = v2_single
        x2_new[start_pos - 1:end_pos + 1] = x2_single
        rewards_v.append(float(reward_v))
        rewards_x.append(float(reward_x))
    return rewards_x, rewards_v, acc2_new, v2_new, x2_new


def DEFAULT_REWARD_FUNCTION(y_pred, y_target):
    # assert y_target.dim() == 1, "y_target must be one-dimensional"
    y_target_new = y_target.reshape(-1)
    sigma = std(y_target_new)
    return 1 / (1 + 1 / sigma * sqrt(mean((y_pred - y_target_new) ** 2)))


def DEFAULT_REWARD_FUNCTION_VELOCITY(a_pred, x_0, v_0, v_obs, dt=0.1):
    v_sim = zeros_like(v_obs)
    v_sim[0] = v_0
    for i in range(1, len(v_obs)):
        v_sim[i] = v_sim[i - 1] + a_pred[i - 1] * dt
    std = std(v_obs)
    return 1 / (1 + 1 / std * sqrt(mean((v_sim - v_obs) ** 2)))


def DEFAULT_REWARD_FUNCTION_TRAJECTORY(a_pred, x_0, v_0, x_obs, dt=0.1):
    v_sim = zeros_like(x_obs)
    x_sim = zeros_like(x_obs)
    v_sim[0] = v_0
    x_sim[0] = x_0
    for i in range(1, len(x_obs)):
        v_sim[i] = v_sim[i - 1] + a_pred[i - 1] * dt
        x_sim[i] = x_sim[i - 1] + v_sim[i - 1] * dt  # + 0.5 * a_pred[i - 1] * dt ** 2

    std = std(x_obs)
    return 1 / (1 + 1 / std * sqrt(mean((x_sim - x_obs) ** 2)))


def simulate(self, x_lead, v_lead, x_init, v_init, opt_values=None, dt=0.5):
    v2_new = zeros(x_lead.shape[0])
    x2_new = zeros(x_lead.shape[0])

    v2_new[0] = v_init
    x2_new[0] = x_init

    length_x1 = x_lead.shape[0]
    # (x, v, delta_v)
    for k in range(1, length_x1):
        X = tensor([x2_new[k - 1] - x_lead[k - 1], v2_new[k - 1], v2_new[k - 1] - v_lead[k - 1]]).reshape(1, -1)
        if opt_values is not None:
            acc = self.execute(X, opt_values['free_const'], opt_values['semi_free_const'])
        else:
            acc = self.execute(X)

        v2_new[k] = v2_new[k - 1] + acc * dt
        x2_new[k] = x2_new[k - 1] + v2_new[k] * dt

    return v2_new, x2_new


def DEFAULT_REWARD_FUNCTION_VELOCITY_TRAJECTORY(prog, x_lead, v_lead, x_obs, v_obs, dt=0.5):
    v2_new = zeros(x_lead.shape[0])
    x2_new = zeros(x_lead.shape[0])
    acc2_new = zeros(x_lead.shape[0])

    v2_new[0] = v_obs[0]
    x2_new[0] = x_obs[0]

    length_x1 = x_lead.shape[0]
    # (x, v, delta_v)
    for k in range(1, length_x1):
        X = tensor([x2_new[k - 1] - x_lead[k - 1], v2_new[k - 1], v2_new[k - 1] - v_lead[k - 1]]).reshape(1, -1)
        acc2_new[k] = prog.execute(X)
        v2_new[k] = v2_new[k - 1] + acc2_new[k] * dt
        x2_new[k] = x2_new[k - 1] + v2_new[k] * dt

    std_x = std(x_obs)
    std_v = std(v_obs)
    reward_x = 1 / (1 + 1 / std_x * sqrt(mean((x2_new - x_obs) ** 2)))
    reward_v = 1 / (1 + 1 / std_v * sqrt(mean((v2_new - v_obs) ** 2)))

    return reward_x, reward_v
