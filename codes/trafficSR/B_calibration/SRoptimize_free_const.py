import os
import datetime

# print("SRfreeconst_start_import", datetime.datetime.now())
from multiprocessing import Pool
import numpy as np
from codes.trafficSR.B_calibration.SRcalibration_func import optimize_free_const

# print("SRfreeconst_end_import", datetime.datetime.now())


def SingleFreeConstOpti(prog_id, progs, free_const_values, semi_free_const_values, X_tensor, y_tensor, bounds,
                        bool_use_rough_calibration=False, ngsim_args=None, ):
    import torch
    # print("Process Id", os.getpid())
    # print("start", prog_id, datetime.datetime.now())
    prog = progs[prog_id]
    for i in range(ngsim_args["n_data_sources"]):
        start_pos = ngsim_args["stop_positions"][i - 1] if i > 0 else 0
        end_pos = ngsim_args["stop_positions"][i]
        init_values = {"init_free_const_values": free_const_values.values[prog_id][i],
                       "init_semi_free_const_values": semi_free_const_values.values[prog_id][i],
                       "bool_use_rough_calibration": bool_use_rough_calibration,
                       "now_data_source_id": i,
                       "bounds": bounds, }
        '''calibrate idm by LBFGS and direct+sqp'''
        savefolder = "pictures/calibration/npy/"

        # method_list = ["LBFGSB", "Direct+SQP"]
        # 2024-8-8 calibration 2-07用来appendix说明迭代步数愈多越精确
        # objective_function_list = ["LOCAL_ACC", "LOCAL_ACC_BOTH"]  # Appendix global_V: "LOCAL_ACC", "GLOBAL_V",

        def calibrate_idm():
            prefix_expression = ["sub", "alpha",
                                 "mul", "alpha",
                                 "add",
                                 "n2", "n2", "div", "v", "v_0",
                                 "n2", "div",
                                 "add", "s_0", "add", "mul", "T", "v",
                                 "div", "mul", "v", "delta_v",
                                 "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                                 "s"]
            name = "idm"
            idm_prog = progs.set_program(prefix_expression)
            for objective_function in objective_function_list:
                for method in method_list:
                    print("------------------calibrate", name, method, objective_function)
                    free_const, semi_free_const, function_value = optimize_free_const(idm_prog, init_values,
                                                                                      X_tensor[
                                                                                      start_pos:end_pos + 1],
                                                                                      y_tensor[
                                                                                      start_pos:end_pos + 1],
                                                                                      ngsim_args=ngsim_args,
                                                                                      method=method,
                                                                                      objective_function=objective_function,
                                                                                      filefolder=savefolder,
                                                                                      name=name)
                    print(free_const, semi_free_const)
                    print()
                    # np.save(savefolder + name + "_" + method + "_" + objective_function + "_free_const.npy",
                    #         free_const.detach().cpu().numpy())

        '''calibrate best formula by LBFGS and direct+sqp'''

        def calibrate_best_formula(name="New Model 3", prefix_expression=None):
            name = name
            prefix_expression = prefix_expression
            best_formula_prog = progs.set_program(prefix_expression)
            for objective_function in objective_function_list:
                for method in method_list:
                    print("------------------calibrate", name, method, objective_function)
                    free_const, semi_free_const, function_value = optimize_free_const(best_formula_prog,
                                                                                      init_values,
                                                                                      X_tensor[
                                                                                      start_pos:end_pos + 1],
                                                                                      y_tensor[
                                                                                      start_pos:end_pos + 1],
                                                                                      ngsim_args=ngsim_args,
                                                                                      method=method,
                                                                                      objective_function=objective_function,
                                                                                      filefolder=savefolder,
                                                                                      name=name)
                    print(free_const, semi_free_const)
                    print()
                    # np.save(savefolder + name + "_" + method + "_" + objective_function + "_free_const.npy",
                    #         free_const.detach().cpu().numpy())

        '''calibrate first-stage and second-stage'''
        # calibrate_idm()
        # prefix_expression = ['sub', 'mul', 'alpha', 'n2', 'sub', 'div', 'v', 'v_0', 'c', 'alpha']
        # calibrate_best_formula(name="New Model 3", prefix_expression=prefix_expression)
        # prefix_expression = ['sub', 'b', 'sub', 'mul', 'alpha', 'sqrt', 'div', 'v', 'v_0', 'div', 'b', 'sub', 'sub',
        #                      'c', 'sub', 'sub',
        #                      'sub', 'div', 'div', 'mul', 'v', 'delta_v', 'add', 'sqrt', 'mul', 'alpha', 'b', 'sqrt',
        #                      'mul', 'alpha', 'b',
        #                      's', 'c', 'c', '1', 'c']
        # calibrate_best_formula(name="New Model 4", prefix_expression=prefix_expression)
        # prefix_expression = ['add', 'div', 'mul', 'v', 'delta_v', 's', 'sub', 'sub', 'mul', 'mul', 'alpha', 'n2', 'sub',
        #                      'div', 'v',
        #                      'v_0', 'c', 'c', 'alpha', 'mul', 'alpha', 'div', 'v', 'v_0']
        # calibrate_best_formula(name="New Model 5", prefix_expression=prefix_expression)
        # return 0., np.zeros(3 + ngsim_args["n_data_sources"] * 2)

        '''normal optimization'''
        objective_function_list = ["LOCAL_ACC"]
        method_list = ["LBFGS", "Direct+SQP"] # LBFGSB
        for objective_function in objective_function_list:
            for method in method_list:
                # if method == "Direct+SQP":
                #     print("------------------calibrate", method, objective_function)
                # function_loss_value是nan
                free_const, semi_free_const, function_loss_value = optimize_free_const(prog, init_values,
                                                                                       X_tensor[
                                                                                       start_pos:end_pos + 1],
                                                                                       y_tensor[
                                                                                       start_pos:end_pos + 1],
                                                                                       ngsim_args=ngsim_args,
                                                                                       method=method,
                                                                                       objective_function=objective_function,
                                                                                       filefolder=None,
                                                                                       name=None)
                if torch.isnan(free_const).any().item() or torch.isnan(
                        semi_free_const).any().item() or np.isnan(function_loss_value):
                    # print("end", prog_id, datetime.datetime.now())
                    return False
                elif function_loss_value >= 0.2:  # no need to fine calibration；0.4 for 跟驰模型
                    break
        free_const_values.values[prog_id][i] = free_const.detach()
        semi_free_const_values.values[prog_id][i] = semi_free_const.detach()
        # print("end", prog_id, datetime.datetime.now())
    return True


def PartitionFreeConstOpti(sub_array, progs, free_const_values, semi_free_const_values, X_tensor, y_tensor, bounds,
                           bool_use_rough_calibration=False, ngsim_args=None, ):
    # print("Process Id", os.getpid())
    # print("start", sub_array, datetime.datetime.now())
    results = []
    for i, prog_id in enumerate(sub_array):
        results.append(
            SingleFreeConstOpti(prog_id, progs, free_const_values, semi_free_const_values, X_tensor, y_tensor, bounds,
                                bool_use_rough_calibration=bool_use_rough_calibration, ngsim_args=ngsim_args, ))
    # print("end", sub_array, datetime.datetime.now())
    return results


def BatchFreeConstOpti(progs, prog_ids, X_tensor, y_tensor, ngsim_args=None, parrallel_mode=False,
                       n_cpus=os.cpu_count()):
    # global expressionID
    lb = progs.library.all_token_info_table.const_lb
    ub = progs.library.all_token_info_table.const_ub
    bounds = list(zip(lb, ub))
    mask_opti_success = np.ones(len(prog_ids), dtype=bool)  # optimize success or not

    free_const_values = progs.free_const_values
    semi_free_const_values = progs.semi_free_const_values
    bool_use_rough_calibration = progs.bool_use_rough_calibration

    if not parrallel_mode or prog_ids.shape[0] < 120:
        for i, prog_id in enumerate(prog_ids):
            mask_opti_success[i] = SingleFreeConstOpti(prog_id, progs, free_const_values,
                                                       semi_free_const_values, X_tensor, y_tensor, bounds,
                                                       bool_use_rough_calibration=bool_use_rough_calibration,
                                                       ngsim_args=ngsim_args)
    else:  # not useful
        mask_include_free_const = progs.library.include_free_const[prog_ids]
        index_include_free_const = np.where(mask_include_free_const)[0]
        pool = Pool(processes=n_cpus)
        results = []
        sub_arrays = np.array_split(prog_ids[index_include_free_const], int(prog_ids.shape[0] / 150) + 1)
        for i, sub_array in enumerate(sub_arrays):
            result = pool.apply_async(PartitionFreeConstOpti, args=(
                sub_array, progs, free_const_values,
                semi_free_const_values,
                X_tensor,
                y_tensor,
                bounds, bool_use_rough_calibration, ngsim_args))
            results.append(result)
        pool.close()
        print("start3", datetime.datetime.now())
        pool.join()
        # print("start4", datetime.datetime.now())
        results = [result.get() for result in results]
        mask_opti_success[index_include_free_const] = np.concatenate(results, axis=0)
    return mask_opti_success
