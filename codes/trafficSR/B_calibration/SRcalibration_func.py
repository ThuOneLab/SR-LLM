import torch
import copy
from codes.trafficSR.B_calibration.calibration_optimizer.SRcalibration_optimizer import lbfgs_optimizer, \
    lbfgsb_optimizer, direct_sqp_optimizer
from codes.trafficSR.C_evaluation.reward_utils.SRreward_function import GLOBAL_SINGLE_TRAJECTORY, use_acc_simulate_v

DEFAULT_OPTIMIZE_DEVICE = "cuda:0"
DEFAULT_OPTIMIZE_DTYPE = torch.float32


# global direct_function, history, global_v_history
# global free_const_history, semi_free_const_history


def MSE_loss(y_pred, y_target):
    # y_pred = torch.tensor(y_pred).to(DEFAULT_OPTIMIZE_DEVICE, DEFAULT_OPTIMIZE_DTYPE)
    # y_target = torch.tensor(y_target).to(DEFAULT_OPTIMIZE_DEVICE, DEFAULT_OPTIMIZE_DTYPE)
    return torch.mean((y_pred - y_target.reshape(-1)) ** 2)


LOSSES = {
    "MSE": MSE_loss
}
OPTIMIZERS = {
    "LBFGS": lbfgs_optimizer,
    "LBFGSB": lbfgsb_optimizer,
    "Direct+SQP": direct_sqp_optimizer,
}
OPJECTIVE_FUNCTIONS = [
    "GLOBAL_X",
    "GLOBAL_V",
    "LOCAL_ACC",
    "LOCAL_ACC_BOTH",
    "LOCAL_ACC_SIM_V"
]


def optimize_free_const(
    prog,
    init_values,
    X,
    y_target,
    ngsim_args=None,
    method="LBFGS",
    loss="MSE",
    objective_function="GLOBAL_V",
    filefolder="pictures/calibration/npy/",
    name=None,
):
    # assert prog.include_free_const, "this program do not need to optimize free const!"
    assert method in OPTIMIZERS.keys(), "unknown optimizer method!"
    assert loss in LOSSES.keys(), "unknown loss type!"
    assert objective_function in OPJECTIVE_FUNCTIONS, "unknown objective function!"

    optimizer = OPTIMIZERS[method]
    loss_func = LOSSES[loss]
    free_const_values = copy.deepcopy(init_values["init_free_const_values"])
    free_const_values = free_const_values.to(X.device)
    semi_free_const_values = copy.deepcopy(init_values["init_semi_free_const_values"])
    semi_free_const_values = semi_free_const_values.to(X.device)
    bool_use_rough_calibration = init_values["bool_use_rough_calibration"]
    now_data_source_id = init_values["now_data_source_id"]
    bounds = init_values["bounds"]

    def objective_func_global_x(free_const, semi_free_const):
        rewards_x, rewards_v, acc2_new, v2_new, x2_new = GLOBAL_SINGLE_TRAJECTORY(
            prog,
            now_data_source_id=now_data_source_id,
            ngsim_args=ngsim_args,
            free_const_value_i=free_const,
            semi_free_const_value_i=semi_free_const,
        )
        start_pos = (
            ngsim_args["stop_positions"][now_data_source_id - 1] + 2
            if now_data_source_id > 0
            else 1
        )
        end_pos = ngsim_args["stop_positions"][now_data_source_id]
        return loss_func(x2_new, ngsim_args["x_obs"][start_pos - 1 : end_pos + 1])

    def objective_func_global_v(free_const, semi_free_const):
        rewards_x, rewards_v, acc2_new, v2_new, x2_new = GLOBAL_SINGLE_TRAJECTORY(
            prog,
            now_data_source_id=now_data_source_id,
            ngsim_args=ngsim_args,
            free_const_value_i=free_const,
            semi_free_const_value_i=semi_free_const,
        )
        start_pos = (
            ngsim_args["stop_positions"][now_data_source_id - 1] + 2
            if now_data_source_id > 0
            else 1
        )
        end_pos = ngsim_args["stop_positions"][now_data_source_id]
        return loss_func(v2_new, ngsim_args["v_obs"][start_pos - 1 : end_pos + 1])

    def objective_func_local_acc_both(free_const, semi_free_const):
        rewards_x, rewards_v, acc2_new, v2_new, x2_new = GLOBAL_SINGLE_TRAJECTORY(
            prog,
            now_data_source_id=now_data_source_id,
            ngsim_args=ngsim_args,
            free_const_value_i=free_const,
            semi_free_const_value_i=semi_free_const,
        )
        start_pos = (
            ngsim_args["stop_positions"][now_data_source_id - 1] + 2
            if now_data_source_id > 0
            else 1
        )
        end_pos = ngsim_args["stop_positions"][now_data_source_id]
        global_v_fun = loss_func(
            v2_new, ngsim_args["v_obs"][start_pos - 1 : end_pos + 1]
        )
        return (
            loss_func(prog.execute(X, free_const, semi_free_const), y_target),
            global_v_fun,
        )

    def objective_func_local_acc(free_const, semi_free_const):
        return loss_func(prog.execute(X, free_const, semi_free_const), y_target)

    def objective_func_local_acc_simulate_v(free_const, semi_free_const):
        acc = prog.execute(X, free_const, semi_free_const)
        start_pos = (
            ngsim_args["stop_positions"][now_data_source_id - 1] + 2
            if now_data_source_id > 0
            else 1
        )
        end_pos = ngsim_args["stop_positions"][now_data_source_id]
        v2_sim = use_acc_simulate_v(
            v0=ngsim_args["v_obs"][start_pos - 1], acc=acc, dt=ngsim_args["dt"]
        )
        return loss_func(v2_sim, ngsim_args["v_obs"][start_pos - 1 : end_pos + 1])

    if objective_function == "GLOBAL_X":
        objective_func = objective_func_global_x
    elif objective_function == "GLOBAL_V":
        objective_func = objective_func_global_v
    elif objective_function == "LOCAL_ACC":
        objective_func = objective_func_local_acc
    elif objective_function == "LOCAL_ACC_BOTH":
        objective_func = objective_func_local_acc_both
    elif objective_function == "LOCAL_ACC_SIM_V":
        objective_func = objective_func_local_acc_simulate_v

    # free_const_optimal_values, semi_free_const_optimal_values, rough_calibration_history = optimizer(
    #     const=torch.cat((free_const_values, semi_free_const_values), dim=0),
    #     free_const_number=len(
    #         free_const_values),
    #     f=objective_func,
    #     bool_use_rough_calibration=True,
    #     bounds=bounds,
    # )
    # free_const_optimal_values, semi_free_const_optimal_values, fine_calibration_history = optimizer(
    #     const=torch.cat((free_const_values, semi_free_const_values), dim=0),
    #     free_const_number=len(
    #         free_const_values),
    #     f=objective_func,
    #     bool_use_rough_calibration=False,
    #     bounds=bounds,)
    # print(prog)
    # print("calibration mse", fine_calibration_history[:, 0])
    # print("calibration time", fine_calibration_history[:, 1])
    '''need slow program'''
    # if rough_calibration_history[-1, 1] > 0.5 and fine_calibration_history[-1, 1] > 0.5:
    #     print(prog)
    #     value = prog.execute(X, free_const_optimal_values, semi_free_const_optimal_values)
    #     print("rough_calibration mse/time", rough_calibration_history[-1, 0], rough_calibration_history[-1, 1])
    #     print("fine_calibration mse/time", fine_calibration_history[-1, 0], fine_calibration_history[-1, 1])
    #     np.save(file="rough_calibration.npy", arr=rough_calibration_history)
    #     np.save(file="fine_calibration.npy", arr=fine_calibration_history)

    '''normal optimization'''
    # history为nan
    free_const_optimal_values, semi_free_const_optimal_values, history = optimizer(
        const=torch.cat((free_const_values, semi_free_const_values), dim=0),
        free_const_number=len(
            free_const_values),
        f=objective_func,
        bool_use_rough_calibration=bool_use_rough_calibration,
        bounds=bounds,
        name=filefolder + name + "_" + method + "_" + objective_function if name is not None else None)
    prog.free_const_values = free_const_optimal_values
    prog.semi_free_const_values = semi_free_const_optimal_values
    return free_const_optimal_values, semi_free_const_optimal_values, history[-1, -1]
