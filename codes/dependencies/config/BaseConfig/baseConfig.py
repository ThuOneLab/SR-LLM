import torch
import numpy as np
# base来源于ngsim_find_idm。并且尽量把参数等都放进去了，到时候直接是字典选取
'''
Used for testing real car trajectories, trajectory files are stored in ./NGSIM_multiVehicle folder
'''

'''MODEL_ARGS固定'''
DEFAULT_MODEL_TYPE = 'combined_lstm'  # 'lstm' or 'transformer'
if DEFAULT_MODEL_TYPE == 'overall_lstm':
    DEFAULT_MODEL_ARGS = {
        "hidden_size": 32,
        "n_layers": 1,
        "is_lobotomized": False,
    }
elif DEFAULT_MODEL_TYPE == 'lstm':
    DEFAULT_MODEL_ARGS = {
        "hidden_size": (128, 128),
        "n_layers": (1, 1),
        "is_lobotomized": False,
    }
elif DEFAULT_MODEL_TYPE == 'embedding_lstm':
    DEFAULT_MODEL_ARGS = {
        "hidden_size": (128, 128),
        "embedding_dim": (32, 32), #之前没有
        "n_layers": (1, 1), #(1, 1)
        "overall_and_partial": (True, True),
        "is_lobotomized": False,
    }
elif DEFAULT_MODEL_TYPE == 'transformer':
    DEFAULT_MODEL_ARGS = {
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 4,
    }
elif DEFAULT_MODEL_TYPE == 'combined_lstm':
    DEFAULT_MODEL_ARGS = {
        "hidden_size": (128, 128),
        "embedding_dim": (256, 256),
        "n_layers": (1, 1), # (2,2)略好，但是时间是四倍
        "overall_and_partial": (False, True),
    }
    
'''OPTIMIZER_ARGS固定'''
DEFAULT_OPTIMIZER_ARGS = {
    "type": 'Adam',
    "lr": 0.0025, # 0.0025
}

'''EARLY_STOP_ARGS统一了，因为总reward是0到1了'''
DEFAULT_EARLY_STOP_ARGS = {
    "stop_reward": 0.99974,#0.9998,
    "stop_after_n_epochs": 1,
}

'''ENV_ARGS有区别'''
DEFAULT_ENV_ARGS = {
    # following are about basic setting
    "device": "cpu", # 竟然继续使用cpu
    "dtype": torch.float32,

    # following are about batch_args
    "batch_size": 1000,
    "max_time_step": 45,

    # following are about loss_args
    'gp_gamma_decay': 0.99,  # 0.7
    'entropy_gamma_decay': 0.9,  # 0.7 0.9(跟驰时)
    'entropy_weight': 0.005,

    # following are parallel
    'parallel_mode': False,
    'n_cpus': 6,  # os.cpu_count()

    # following are reward_args
    'risk_factor': 0.1,  # 0.05
    "similarity_args": {
        "compute_similarity": True,
        "similarity_compute_limit": 0.,
        "similarity_reward": None,
    },
    "reward_weight": {
        "factor_complexity": 0.025,
        "factor_similarity": 0.075,
        "factor_rmse": 0.9,
    }
}

'''PRIOR_ARGS有区别'''
DEFAULT_PRIOR_ARGS = {
    "prior_type": [
        "HardLength",
        # "SoftLength",  # 只有关于min的约束，从而能够更好地让其增长——在跟驰或者物理的时候不好用
        # will have fewer elites, but higher quality; but when too complex, don't use
        # if find old expressions, don't use it这个被改造成了min
        "SoftMaxLength",
        # "Arity", # equal to max length judge
        "Const", # physo最佳公式没用到 目前改成了只关于1op运算符的限制
        # "ConstOnce", # not useful
        "NoneSingleArityInverse",
        "NoneDoubleArityInverse",
        "NoneDoubleArityOffset", # PhySO没有
        "NoneDoubleAritySymmetry", # PhySO没有
        # "NoneTranscendNest", # 不够好,不如NoneNested
        # "NoneFreeConstRisky" # maybe not useful
        "PhysicalUnits",
        "NoneDoubleConst", # physo最佳公式没用到 add c3 c3
        # "Power", # 是在有pow符号的时候才会用到，变量 常量
        # "FirstNotFourOperators", # 会造成一些可能不好的搜索空间减小的结果
        "NoneNested"
    ],
    "prior_config": {
        # Length RELATED
        "LengthPrior": {"min_length": 4, "max_length": None, },
        "SoftLengthPrior": {"length_loc": 32,"max_length_loc": 32, "scale": 30, "eps": 1e-2}, # length_loc是期望的长度，scale是标准差（越大高斯函数下降越平缓）——可以取为length_loc/1.3
        # RELATIONSHIPS RELATED
        "PhysicalUnitsPrior": {"prob_eps": np.finfo(np.float32).eps},
        # Operators RELATED
        "FirstNotFourOperatorsPrior":{"NotFour": False},
    }
}

'''TOKEN_ARGS有区别'''
DEFAULT_TOKEN_ARGS = {
    # following are about token_args
    "operator_tokens": ["add", "mul", "sub", "div", "n2", "sqrt"],
    "operator_description": {
        "add": "addition",
        "mul": "multiplication",
        "sub": "subtraction",
        "div": "division",
        "n2": "square",
        "n3": "cube",
        "n4": "fourth power",
        "sqrt": "square root",
        "sin": "sine",
        "cos": "cosine",
        "tan": "tangent",
        "exp": "exponential",
        "log": "logarithm",
        "pow": "power",
        "abs": "absolute value",
        "neg": "negative",
        "inv": "inverse", # pow sin cos tan abs
    },
    
    "variable_tokens": ["s", "v", "delta_v", ],
    "variable_description": {
        "s": "The symbol in distance unit which represents the current following distance",
        "v": "The symbol in speed unit which represents ego vehicle speed",
        "delta_v": "The symbol in speed unit which represents subtracting the speed of the preceding vehicle from the speed of ego vehicle",
        # for GHR and Helly
        "delta_x": "The symbol in distance unit which represents subtracting the position of the ego vehicle from the position of preceding vehicle",
        "delta_vi": "The symbol in speed unit which represents subtracting the speed of the ego vehicle from the speed of preceding vehicle", #代表是delta_v的inverse形式
        # for easy:最多五个变量
        "x_1": "The symbol 1 in dimensionless unit",
        "x_2": "The symbol 2 in dimensionless unit",
        "x_3": "The symbol 3 in dimensionless unit",
        "x_4": "The symbol 4 in dimensionless unit",
        "x_5": "The symbol 5 in dimensionless unit",
    },
    "variable_units": {
        "s": [1, 0],
        "v": [1, -1],
        "delta_v": [1, -1],
        "delta_x": [1, 0],
        "delta_vi": [1, -1],
        "x_1": [0, 0],
        "x_2": [0, 0],
        "x_3": [0, 0],
        "x_4": [0, 0],
        "x_5": [0, 0],
    },
    
    "free_const_tokens": ["alpha", "b", "v_0", "T", "s_0", "c", "k_1", "k_2"],
    "free_const_description": {
        # for IDM
        "alpha": "The symbol in acceleration unit which represents the comfortable maximum acceleration of ego vehicle",
        "b": "The symbol in acceleration unit which represents comfortable maximum deceleration for the driver",
        "v_0": "The symbol in speed unit which represents desired ego vehicle speed",
        "T": "The symbol in time unit which represents the desired time headway",
        "s_0": "The symbol in distance unit which represents the desired minimum following distance in stationary state",
        # for GHR
        "c": "The symbol in dimensionless unit which represents the overall strength of the response",
        # for Helly
        "Beta": "The symbol in time unit which represents the leveling factor multiplied by the velocity factor",
        "k_1": "The symbol in frequency unit which represents the sensitivity of the following vehicle to the speed difference between itself and the leading vehicle",
        "k_2": "The symbol in productFrequency unit which represents the sensitivity of the following vehicle to the distance difference between itself and the leading vehicle",
        # for easy
        "c_1": "free constant 1",
        "c_2": "free constant 2",
        "c_3": "free constant 3",
        "c_4": "free constant 4",
        "c_5": "free constant 5",
        "c_6": "free constant 6",
        "c_7": "free constant 7",
        "c_8": "free constant 8",
        "c_9": "free constant 9",
        "c_10": "free constant 10",
        # for Fermann
    },
    "free_const_initial_values_dict": {
        # for IDM
        "alpha": 2.0,
        "b": 2.5,
        "v_0": 20.,
        "T": 1.5,
        "s_0": 5.5,
        # for GHR
        "c": 1.05,
        # for Helly
        "Beta": 1.05,
        "k_1":0.55,
        "k_2": 0.55,
        # for easy
        "c_1": 1.,
        "c_2": 1.,
        "c_3": 1.,
        "c_4": 1.,
        "c_5": 1.,
        "c_6": 1.,
        "c_7": 1.,
        "c_8": 1.,
        "c_9": 1.,
        "c_10": 1.,
        # for Fermann
    },
    "free_const_units": {
        "alpha": [1, -2],
        "b": [1, -2],
        "v_0": [1, -1],
        "T": [0, 1],
        "s_0": [1, 0],
        "c": [0, 0],
        "Beta": [0, 1],
        "k_1": [0, -1],
        "k_2": [0, -2],
        "c_1": [0, 0],
        "c_2": [0, 0],
        "c_3": [0, 0],
        "c_4": [0, 0],
        "c_5": [0, 0],
        "c_6": [0, 0],
        "c_7": [0, 0],
        "c_8": [0, 0],
        "c_9": [0, 0],
        "c_10": [0, 0],
    },
    "free_const_bounds": {
        "alpha": [1, 3],
        "b": [1, 4],
        "v_0": [10, 30],
        "T": [0, 3],
        "s_0": [1, 10],
        "c": [0.1, 2.],
        "Beta": [0.1, 2.],
        "k_1": [0.1, 1],
        "k_2": [0.1, 1],
        "c_1": [-10, 10],
        "c_2": [-10, 10],
        "c_3": [-10, 10],
        "c_4": [-10, 10],
        "c_5": [-10, 10],
        "c_6": [-10, 10],
        "c_7": [-10, 10],
        "c_8": [-10, 10],
        "c_9": [-10, 10],
        "c_10": [-10, 10],
    },
    
    "semi_free_const_tokens": [],
    "semi_free_const_description": {},
    "semi_free_const_initial_values_dict": {},
    "semi_free_const_units": {},
    "semi_free_const_bounds": {},
    
    "fixed_const_tokens": ["1"],
    "fixed_const_values_dict": {"1": 1.,"s_20": 20.,"pi":np.pi,"2*pi":2*np.pi,"4*pi":4*np.pi},
    "fixed_const_units": {"1": [0, 0],"s_20": [1, 0],"pi":[0,0],"2*pi":[0,0],"4*pi":[0,0]},
    "fixed_const_description": {"1": "constant 1",
                                "s_20": "distance constant 20m",
                                "pi": "constant pi",
                                "2*pi": "constant 2*pi",
                                "4*pi": "constant 4*pi"},
    
    "combination_tokens": [
        # "factor_v_ratio",
        # "sin(x)",
        # "sin(x^2)",
        # "cos(x)",
        # "sin(theta2)",
    ],
    # "combination_infix_expression": [
    #     # "factor_v_ratio",
    # ],
    # "combination_infix_expression": {
    #     "factor_v_ratio":"factor_v_ratio",
    # },
    "combination_description": {
        "factor_v_ratio": "The symbol in dimensionless unit which represents the influence of speed ratio on the acceleration",
        "sin(x)": "The symbol in dimensionless unit which represents the sine of x",
        "sin(x^2)": "The symbol in dimensionless unit which represents the sine of x^2",
        "cos(x)": "The symbol in dimensionless unit which represents the cosine of x",
        "sin(theta2)": "The symbol in dimensionless unit which represents the sine of theta2",
    },
    "combination_units": {
        "factor_v_ratio": [0, 0],
        "sin(x)": [0, 0],
        "sin(x^2)": [0, 0],
        "cos(x)": [0, 0],
        "sin(theta2)": [0, 0],
    },
    "combination_prefix_expression": {
        "factor_v_ratio": ["div", "v", "v_0"],
        "sin(x)": ["sin", "x_1"],
        "sin(x^2)": ["sin", "n2", "x_1"],
        "cos(x)": ["cos", "x_1"],
        "sin(theta2)": ["sin", "theta2"],
    }
}

'''LIBRARY_ARGS有区别'''
DEFAULT_LIBRARY_ARGS = {
    "tokens_args": DEFAULT_TOKEN_ARGS,
    "min_occupancy_times":{"targets_name":[],"min_times":[],},
    "superparent_units": [1, -2],
    "superparent_names": ["IDM", "GHR", "Helly"],
    "superparent_prog": [[]],
}

'''SAC_VALUE_ARGS固定'''
DEFAULT_SAC_VALUE_ARGS = {
    "hidden_size": (256, 256),
    "n_layers": (1, 1),
    "overall_and_partial": (False, True),
}

'''SAC_ARGS固定'''
DEFAULT_SAC_ARGS = {
    "actor_lr": 3e-4,  # 1e-3,0.0005, 1e-3
    "critic_lr": 1e-3,  # 1e-2,0.0007,1e-2, 3e-4
    "alpha_lr": 3e-4,  # 1e-2,1e-3,1e-3
    # "target_entropy": -1,
    "tau": 0.05, # 别人取这个值0.01
    "gamma": 0.99,  # discount,0.99
    "n_warmup_batches": 1, # 8,10
    "sample_batch_size": DEFAULT_ENV_ARGS["batch_size"] * 5,  # 15,12过大会取12000个,8
    "replay_buffer_capacity": int(DEFAULT_ENV_ARGS['risk_factor']*DEFAULT_ENV_ARGS["batch_size"] * DEFAULT_ENV_ARGS["max_time_step"] /3 * 5), # DEFAULT_ENV_ARGS['risk_factor']*DEFAULT_ENV_ARGS["batch_size"] * DEFAULT_ENV_ARGS["max_time_step"]是进入的状态数量，/3是比较实际的非end状态数量，*x是为了保证能够取到足够的x轮数据
    #1000000过大
}

'''AGENT_ARGS有区别'''
DEFAULT_AGENT_ARGS = {
    "fewshot_num": 3,
    "best_symbol_num": 2,  # 2 3_not_good, 3 for idm(only directly built)
    "delete_combination_num": 2,  # same as best_symbol_add_num
    "best_expression_num": 5,
    "reflection_num": 1,
    "extend_num": 1,
    "max_try_num": 3,
}

'''BOOL_ARGS有区别'''
DEFAULT_BOOL_ARGS = {
    # Data Setting: 如果是NGSIM的true trajectory, 则需要设置为True，会导致reward_function使用仿真法
    "bool_use_true_trajectory": True,

    # DRL Epoch Setting
    "n_epochs": 30, # need change
    "epochs_save_expression_number": 15,
    "bool_use_UCB_for_sampling": True, # UCB适合于需要incremental探索的情况，比如迭代式的学习跟驰公式

    # plot picture
    "bool_plot_intermediate_process": True,
    "plot_intermediate_process_limit": 0.95,  # plot following picture
    
    # DRL_PPO settings: multiEnvs settings
    "n_workers": 1,

    # DRL experience replay: old and not useful
    "bool_DRL_experience_replay": False,
    "experience_replay_rate": 0.02,
    "retain_data_rounds": 3,  # 2 3 5 7
    "exp_decay_epsilon_greedy_setting": {
        "decay_ratio": 0.99,
        "init_epsilon": 1.0,
        "min_epsilon": 0.5},
    # Calibration Setting: old and no use now
    "bool_use_rough_calibration": False,
    "calibration_conversion_limit": 0.6,

    # LLM Setting About Evaluation
    "bool_select_k_combinations": True,  # update(and evaluation), if need update, then start evaluation
    "n_selection_of_combinations": 8,
    "bool_compute_numerical_score": True,  # numerical_score
    "bool_compute_semantic_score": True,  # semantic_score

    # use combinations or not
    "bool_use_combinations": True, #用于判定是否要使用combinations
    
    # single learning or incremental
    "bool_use_evolutions": True,
    "n_evolutions": 30,
    
    # LLM Setting About Extract
    "bool_extract_valuable": False,  # extract: old method
    "bool_agent_single_step_increase": True,  # single increment by agent: new method
    "new_tokens_number": 10,

    # LLM Setting About Reflection
    "bool_reflection": False,

    # LLM Setting About Explain: old and no use now
    "bool_explain_final_expressions": False,  # LLM explain

    # find old formula or new formula
    "bool_find_new_formula": False,
    
    # 是否是Feynman
    "bool_is_feynman": False,
    
    # 是否是Random
    "bool_is_random": False,
}