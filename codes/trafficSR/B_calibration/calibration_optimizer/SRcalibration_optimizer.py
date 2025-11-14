import time
import numpy as np
import torch
import copy
from scipy.optimize import direct
from scipy.optimize import minimize
from codes.trafficSR.B_calibration.calibration_optimizer.SRlbfgs import LBFGS

DEFAULT_LBFGS_ARGS = {
    'n_steps': 5, #15,physo是15
    'tol': 1e-8, #1e-6.
    'lbfgs_func_args': {
        'lr': 1,  # for x/v calibration, normally 1; 没有用上
        'max_iter': 4, # 和physo一样
        'line_search_fn': "strong_wolfe",
    },
}
DEFAULT_LBFGSB_ARGS = {
    'lbfgsb_func_args': {
        "method": "L-BFGS-B",
        'tol': 1e-4,
        # optional Tolerance for termination. if CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH, make it smaller
        'options': {
            'maxiter': 3, # 3,10
            'eps': 1e-4,  # 1e-8 is too small, then grad will be 0 leading to convergence.
            # 'disp': True,
            'disp': False,
            'maxls': 10,  # Line search cannot locate an adequate point after MAXLS
            # function and gradient evaluations.
        },
        # 'line_search_fn': "strong_wolfe",
    },
}
DEFAULT_DIRECT_SQP_ARGS = {
    # 'n_steps': 30,
    'direct_args': {
        'eps': 1e-4, #before 3-3:1e-2,  # before 5-29:1e-4
        "maxiter": 20,#before 3-3:200,
        "f_min": 0.,
        "f_min_rtol": 0.01,
        'vol_tol': 1e-16, # before 3-3:1e-4,  # 1e-16 # before 5-29:1e-6
    },
    'sqp_args': {
        'method': 'SLSQP',
        'tol': 1e-16, #before 3-3:1e-6,
        'options': {
            'eps': 1e-4,
            'maxiter': 400,
            # 'disp': True,
            'disp': False,
        },
    }
}


def lbfgs_optimizer(const, free_const_number, f, bool_use_rough_calibration=True, args=DEFAULT_LBFGS_ARGS,
                    bounds=None, name=None):
    eps = 1e-6
    n_steps = args['n_steps']
    tol = args['tol']
    lbfgs_func_args = args['lbfgs_func_args']

    # params = torch.tensor(params).to(DEFAULT_OPTIMIZE_DEVICE, DEFAULT_OPTIMIZE_DTYPE)
    original_const = copy.deepcopy(const)
    if bool_use_rough_calibration:
        const = const[:free_const_number]
        const.requires_grad = True
        lbfgs = LBFGS([
            {'params': const}], **lbfgs_func_args, bounds=bounds)
    else:
        # lbfgs = torch.optim.LBFGS([
        #     {'params': free_const},
        #     {'params': semi_free_const}], **lbfgs_func_args) # can not be group
        const.requires_grad = True
        lbfgs = LBFGS([
            {'params': const}], **lbfgs_func_args, bounds=bounds)

    time_history = [0.]
    history = []
    if bool_use_rough_calibration:
        history.append(f(const, original_const[free_const_number:]).item())
    else:
        history.append(f(const[:free_const_number], const[free_const_number:]).item())

    def closure():
        lbfgs.zero_grad()
        if bool_use_rough_calibration:
            objective = f(const, original_const[free_const_number:])
        else:
            objective = f(const[:free_const_number], const[free_const_number:])
        # history.append(objective.item())
        objective.requires_grad_(True)
        objective.backward()
        return objective

    start_time = time.time()
    last_mse = 0.
    for i in range(n_steps):
        lbfgs.step(closure)
        if bool_use_rough_calibration:
            history.append(f(const, original_const[free_const_number:]).item())
        else:
            history.append(f(const[:free_const_number], const[free_const_number:]).item())
        end_time = time.time()
        time_history.append(end_time - start_time)
        if torch.any(torch.isnan(const)):  # if nan, break
            # print("nan in const!")
            break
        if history[-1] < tol or abs(history[-1] - last_mse) < eps:  # if converge, break
            # print("converge!")
            break
        last_mse = history[-1]
    # print("final fun", history[-1])
    # print("final time", time.time() - start_time)
    return_value = np.stack((time_history, history), axis=1)
    if name is not None:
        np.save(name + "_time_history.npy", return_value)
    if bool_use_rough_calibration:
        return const, original_const[free_const_number:], return_value
    else:
        return const[:free_const_number], const[free_const_number:], return_value


def lbfgsb_optimizer(const, free_const_number, f, bool_use_rough_calibration=True, args=DEFAULT_LBFGSB_ARGS,
                     bounds=None, name=None):
    def direct_fun(theta):
        value = f(torch.tensor(theta), torch.tensor([]))
        if type(value) is tuple:
            return value[0].item()
        else:
            return value.item()

    value_result = f(const, torch.tensor([]))
    if type(value_result) is tuple:
        history = [value_result[0].item()]
        global_v_history = [value_result[1].item()]
    else:
        history = [value_result.item()]
        global_v_history = []
    free_const_history = [const[:free_const_number].numpy()]
    time_history = [0.]
    start_time = time.time()
    initial_guess = const

    def callback_lbfgsb(xk):
        # print("lbfgsb_Iteration: {}, X: {}".format(callback_lbfgsb.iteration, xk))
        callback_lbfgsb.iteration = callback_lbfgsb.iteration + 1
        lbfgsb_time = time.time()
        time_history.append(lbfgsb_time - start_time)
        result = f(torch.tensor(xk), torch.tensor([]))
        if type(result) is tuple:
            history.append(result[0].item())
            global_v_history.append(result[1].item())
        else:
            history.append(result.item())
        free_const_history.append(xk)

    callback_lbfgsb.iteration = 0
    optimal_solution = minimize(direct_fun, initial_guess.numpy(), **args['lbfgsb_func_args'], bounds=tuple(bounds),
                                callback=callback_lbfgsb)
    # print("result_fun, result_nfev", optimal_solution.fun, optimal_solution.nfev)
    # print("final result of lbfgsb", optimal_solution.x)
    # print("final time", time.time() - start_time)
    if len(global_v_history) > 0:
        return_value = np.stack((time_history, history, global_v_history), axis=1)
    else:
        return_value = np.stack((time_history, history), axis=1) # time, mse_value
    if name is not None:
        np.save(name + "_time_history.npy", return_value)
        np.save(name + "_free_const_history.npy", np.array(free_const_history))
    return_const = torch.tensor(optimal_solution.x)
    return return_const[:free_const_number], return_const[free_const_number:], return_value


def direct_sqp_optimizer(const, free_const_number, f, bool_use_rough_calibration=True, args=DEFAULT_DIRECT_SQP_ARGS,
                         bounds=None, name=None):
    def direct_fun(theta):
        value = f(torch.tensor(theta), torch.tensor([]))
        if type(value) is tuple:
            return value[0].item()
        else:
            return value.item()

    value_result = f(const, torch.tensor([]))
    if type(value_result) is tuple:
        history = [value_result[0].item()]
        global_v_history = [value_result[1].item()]
    else:
        history = [value_result.item()]
        global_v_history = []
    free_const_history = [const[:free_const_number].numpy()]
    time_history = [0.]
    start_time = time.time()

    def callback_direct(xk):
        # print("DIRECT_Iteration: {}, X: {}".format(callback_direct.iteration, xk))
        callback_direct.iteration = callback_direct.iteration + 1
        direct_time = time.time()
        time_history.append(direct_time - start_time)
        result = f(torch.tensor(xk), torch.tensor([]))
        if type(result) is tuple:
            history.append(result[0].item())
            global_v_history.append(result[1].item())
        else:
            history.append(result.item())
        free_const_history.append(xk)
        # print("used time", direct_time - start_time)

    callback_direct.iteration = 0
    optimal_solution = direct(direct_fun, bounds, **args['direct_args'], callback=callback_direct)
    initial_guess = optimal_solution.x

    # print("result_fun, result_nfev", optimal_solution.fun, optimal_solution.nfev)
    # print("initial_guess of sqp", initial_guess)

    def callback_sqp(xk):
        # print("SQP_Iteration: {}, X: {}".format(callback_sqp.iteration, xk))
        callback_sqp.iteration = callback_sqp.iteration + 1
        sqp_time = time.time()
        time_history.append(sqp_time - start_time)
        result = f(torch.tensor(xk), torch.tensor([]))
        if type(result) is tuple:
            history.append(result[0].item())
            global_v_history.append(result[1].item())
        else:
            history.append(result.item())
        free_const_history.append(xk)
        # print("used time", sqp_time - start_time)

    callback_sqp.iteration = 0
    optimal_solution = minimize(direct_fun, initial_guess, **args['sqp_args'], bounds=tuple(bounds),
                                callback=callback_sqp)
    # print("result_fun, result_nfev", optimal_solution.fun, optimal_solution.nfev)
    # print("final result of sqp", optimal_solution.x)
    # print("final time", time.time() - start_time)
    if len(global_v_history) > 0:
        return_value = np.stack((time_history, history, global_v_history), axis=1)
    else:
        return_value = np.stack((time_history, history), axis=1)
    if name is not None:
        np.save(name + "_time_history.npy", return_value)
        np.save(name + "_free_const_history.npy", np.array(free_const_history))
    return_const = torch.tensor(optimal_solution.x)
    return return_const[:free_const_number], return_const[free_const_number:], return_value
