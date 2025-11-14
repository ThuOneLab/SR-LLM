if __name__ == '__main__':
    import os
    # print("当前目录os.getcwd():",os.path.abspath(os.getcwd()))
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from codes.trafficSR.E_train_codes.RL_SAC_LLM.RL_SAC_LLM_manual import RL_LLM_train as RL_LLM_train_manual
    from codes.trafficSR.E_train_codes.RL_SAC_LLM.RL_SAC_LLM import RL_LLM_train as RL_LLM_train
    from codes.trafficSR.utils import *
    from codes.dependencies.config.FindNewFormula.ngsim_find_new import DEFAULT_TRAIN_ARGS as NEW_CONFIG
    import numpy as np

    setup_seed(myseed)
    filter_dt = 0.5
    '''single traj'''
    # X_array, y_array, ngsim_args, data_source, following_ID, preceeding_ID = ngsim_single_trajectory(
    #     filter_dt=filter_dt)
    X_array, y_array, ngsim_args, data_source, following_ID, preceeding_ID = ngsim_single_trajectory(
        filter_dt=filter_dt, file_folder="data/NGSIM_dataNpy/trajectories-0820am-0835am",
        filename="ego17_preceeding13.npy")
    # RL_LLM_train(X=X_array, y=y_array, train_args=NGSIM_CONFIG, ngsim_args=ngsim_args,
    #                                              data_source=data_source)

    '''filefolder traj'''
    # "NGSIM_multiVehicle"
    # "data/NGSIM_dataNpy/trajectories-0750am-0805am"
    # "data/NGSIM_dataNpy/trajectories-0805am-0820am"
    # find new formula
    X_array, y_array, ngsim_args, data_source = ngsim_filefolder_trajectory(filter_dt=filter_dt, filefolder_list=[
        "data/NGSIM_dataNpy/trajectories-0820am-0835am"], need_first_K=20)

    '''find new model'''
    X_array, y_array, ngsim_args, data_source, following_ID, preceeding_ID = ngsim_single_trajectory(
        filter_dt=filter_dt, file_folder="data/NGSIM_dataNpy/trajectories-0820am-0835am",
        filename="ego17_preceeding13.npy", data_process_algorithm=data_process_IDM)
    R_LAST_EPOCH,_,_ = RL_LLM_train(X=X_array, y=y_array, train_args=NEW_CONFIG, ngsim_args=ngsim_args,
                                data_source=data_source,
                                seed=myseed, memory_path="codes/ragLibrary/memory_ngsim_find_new")

    np.save("pictures/E_global_analysis/plot_last_epoch_of_incremental/reward_list_last_epoch.npy", R_LAST_EPOCH)
