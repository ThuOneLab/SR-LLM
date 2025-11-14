import torch
import numpy as np

DEFAULT_MODEL_TYPE = 'embedding_lstm'  # 'lstm' or 'transformer'

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
        "hidden_size": (64, 64),
        "n_layers": (1, 1),
        "overall_and_partial": (True, True),
        "is_lobotomized": False,
    }

elif DEFAULT_MODEL_TYPE == 'transformer':
    DEFAULT_MODEL_ARGS = {
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 4,
    }

DEFAULT_SAC_VALUE_ARGS = {
    "hidden_size": (64, 64),
    "n_layers": (1, 1),
    "overall_and_partial": (True, True),
}

DEFAULT_OPTIMIZER_ARGS = {
    "type": 'Adam',
    "lr": 0.005,
}

DEFAULT_PRIOR_ARGS = {
    "prior_type": [
        "HardLength",
        # "SoftLength",  # need: will have fewer elites, but higher quality; but when too complex, don't use
        # if find old expressions, don't use it
        "SoftMaxLength",
        # "Arity", # equal to max length judge
        "Const",
        # "ConstOnce", # not useful
        "NoneSingleArityInverse",
        "NoneDoubleArityInverse",
        "NoneDoubleArityOffset",
        "NoneDoubleAritySymmetry",
        "NoneTranscendNest",
        # "NoneFreeConstRisky" # maybe not useful
        "PhysicalUnits",
        "NoneDoubleConst",
    ],
    "prior_config": {
        "LengthPrior": {"min_length": 4, "max_length": None, },
        "SoftLengthPrior": {"length_loc": 32, "scale": 30, "eps": 1e-2},  # 32_26 is best
        # RELATIONSHIPS RELATED
        "PhysicalUnitsPrior": {"prob_eps": np.finfo(np.float32).eps},  # PHYSICALITY
    }
}

DEFAULT_EARLY_STOP_ARGS = {
    "stop_reward": 2.0 - 1e-5,
    "stop_after_n_epochs": 5,
}

DEFAULT_ENV_ARGS = {
    "device": "cpu",
    "dtype": torch.float32,

    # following are about batch_args
    "batch_size": 1000,
    "max_time_step": 40,

    # following are about loss_args
    'gp_gamma_decay': 0.99,  # 0.7
    'entropy_gamma_decay': 0.9,  # 0.7
    'entropy_weight': 0.005,

    # following are parallel
    'parallel_mode': True,
    'n_cpus': 6,  # os.cpu_count()

    # following are reward_args
    'risk_factor': 0.1,  # 0.05
    "similarity_args": {
        "compute_similarity": True,
        "similarity_compute_limit": 0.,
        "similarity_reward": None,
    },
    "reward_weight": {
        "factor_complexity": 0.,
        "factor_similarity": 0.2,
        "factor_rmse": 0.8,
    }
}

new_tokens = [
    {
        "token_name": "T",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "desired time headway",
        "physical_units": {
            "m": 0,
            "s": 1
        }
    },
    {
        "token_name": "a_lead",
        "is_basic": True,
        "token_type": "input_variable",
        "length": 1,
        "description": "acceleration of the leading vehicle",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "a_comfort",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "comfortable acceleration level for the driver",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "delta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "exponent for speed sensitivity",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "Gamma",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "exponent for following distance sensitivity",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "theta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "comfortable maximum acceleration of the following car",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "epsilon",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "coefficient for environmental influence",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "environment",
        "is_basic": True,
        "token_type": "input_variable",
        "length": 1,
        "description": "environmental conditions affecting driving behavior",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "beta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "parameter for adjusting driver behavior",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "rho",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "traffic density",
        "physical_units": {
            "m": -1,
            "s": 0
        }
    }
]

DEFAULT_TOKEN_ARGS = {
    # following are about token_args
    "operator_tokens": ["add", "mul", "sub", "div", "n2", "sqrt", ],
    "operator_description": {
        "add": "addition",
        "mul": "multiplication",
        "sub": "subtraction",
        "div": "division",
        "n2": "square",
        "sqrt": "square root",
        # "inv": "inverse",
        # "pow": "power",
    },
    "variable_tokens": ["s", "v", "delta_v", ],  # "delta_v",
    "variable_description": {
        # "s": "following distance(m)",
        # "v": "ego vehicle speed(m/s)",
        # "delta_v": "The speed difference between the front and the ego(m/s)",
        # "a_lead": "acceleration of the leading vehicle(m/s^2)",
        "s": "The symbol in distance unit which represents the current following distance",
        "v": "The symbol in speed unit which represents ego vehicle speed",
        "delta_v": "The symbol in speed unit which represents subtracting the speed of the preceding vehicle from the speed of ego vehicle",
    },
    "variable_units": {
        "s": [1, 0],
        "v": [1, -1],
        "delta_v": [1, -1],
        # "a_lead": [1, -2],
    },

    "free_const_tokens": ["alpha", "b", "v_0", "T", "s_0", ],
    "free_const_description": {
        # "alpha": "comfortable maximum acceleration of the following vehicle(m/s^2)",
        # "b": "comfortable deceleration level for the driver(m/s^2)",
        # "v_0": "desired ego vehicle speed(m/s)",
        # "T": "desired time headway(s)",
        # "s_0": "desired safe distance(m)",
        # "k1": "A free constant that can be used for adjusting the model",
        # "k2": "A free constant that can be used for adjusting the model",
        # "k3": "A free constant that can be used for adjusting the model",
        "alpha": "The symbol in acceleration unit which represents the comfortable maximum acceleration of ego vehicle",
        "b": "The symbol in acceleration unit which represents comfortable maximum deceleration for the driver",
        "v_0": "The symbol in speed unit which represents desired ego vehicle speed",
        "T": "The symbol in time unit which represents the desired time headway",
        "s_0": "The symbol in distance unit which represents the desired minimum following distance in stationary state",
    },
    # "free_const_initial_values": [2., 2.5, 20., 1.5, 5.5, 0., 0., 0.],
    # "free_const_initial_values": [1., 1., 10., 0.5, 2.],
    # "free_const_initial_values": [1.5, 1.5, 15., 0.5, 2.],
    "free_const_initial_values": [2., 2.5, 20., 1.5, 5.5, ],
    # "free_const_initial_values": [2., 20., 1.5, 5.5, ],
    "free_const_units": {
        "alpha": [1, -2],
        "b": [1, -2],
        "v_0": [1, -1],
        "T": [0, 1],
        "s_0": [1, 0],
        # "k1":[0, 0],
        # "k2":[0, 0],
        # "k3":[0, 0],
    },
    "free_const_bounds": {
        "alpha": [1, 3],
        "b": [1, 4],
        "v_0": [10, 30],
        "T": [0, 3],
        "s_0": [1, 10],
        # "k1":[-100,100],
        # "k2":[-100,100],
        # "k3":[-100,100],
    },

    # "semi_free_const_tokens": ["a_comfort", "delta", "Gamma", ],
    # "semi_free_const_description": {
    #     "a_comfort": "comfortable acceleration level for the driver",
    #     "delta": "exponent for speed sensitivity",
    #     "Gamma": "exponent for following distance sensitivity",
    # },
    # "semi_free_const_initial_values": [1., 1., 1.],
    # "semi_free_const_units": {
    #     "a_comfort": [1, -2],
    #     "delta": [0, 0],
    #     "Gamma": [0, 0],
    # },
    "semi_free_const_tokens": [],
    "semi_free_const_description": {

    },
    "semi_free_const_initial_values": [],
    "semi_free_const_units": {

    },
    "semi_free_const_bounds": {

    },
    "fixed_const_tokens": ["1", ],
    "fixed_const_values": [1., ],
    "fixed_const_units": {"1": [0, 0]},
    "fixed_const_description": {"1": "constant 1"},
    # "fixed_const_tokens": ["1", "2"],
    # "fixed_const_values": [1., 2.],
    # "fixed_const_units": {"1": [0, 0],
    #                       "2": [0, 0]},
    # "fixed_const_description": {"1": "constant 1",
    #                             "2": "constant 2"},
    "combination_tokens": [
        "v / v0", "s0 / s", "b * T", "T * v", "v * delta_v", "sqrt(alpha * b)",
        "s_safe",
        "a_v_square",
        "factor_s",
        "factor_v_square",
        "factor_s_square",
    ],
    "combination_infix_expression": [
        "v / v0",
        "s0 / s",
        "b * T",
        "T * v",
        "v * delta_v",
        "sqrt(alpha * b)",
        "a_lead / alpha",
        "v^delta/a_comfort",
        "delta_v^delta",
        "s^Gamma",
        "s_0+v*T+v*delta_v/(sqrt(alpha*b)+sqrt(alpha*b))",
        "alpha-alpha*(v/v_0)^2",
        "s_safe",
        "a_v_square",
        "factor_s",
        "factor_v_square",
        "factor_s_square",
    ],
    "combination_description": {
        "v / v0": "ratio of current speed to desired speed",
        "s0 / s": "ratio of current following distance to desired safe distance",
        "b * T": "product of comfortable deceleration and time headway",
        "T * v": "product of time headway and ego vehicle's speed",
        "v * delta_v": "product of ego vehicle's speed and speed difference",
        "sqrt(alpha * b)": "square root of the product of comfortable maximum acceleration and comfortable deceleration",
        "a_lead / alpha": "ratio of leading vehicle's acceleration to comfortable maximum acceleration",
        "v * delta_v / a_comfort": "ratio of ego vehicle's speed times speed difference to comfortable acceleration level, only active in non-stationary flow",
        "v^delta/a_comfort": "ratio of ego vehicle's speed raised to the power of speed sensitivity to comfortable acceleration level",
        "delta_v^delta": "speed difference raised to the power of speed sensitivity exponent",
        "s^Gamma": "following distance raised to the power of following distance sensitivity exponent",
        "s_safe": "The symbol in distance unit which represents the safe following distance",
        "a_v_square": "The symbol in acceleration unit which represents the square of the speed of ego vehicle",
        "factor_s": "The symbol in dimensionless unit which represents the influence of distance",
        "factor_v_square": "The symbol in dimensionless unit which represents the influence of speed square",
        "factor_s_square": "The symbol in dimensionless unit which represents the influence of distance square",
    },
    "combination_units": {
        "v / v0": [0, 0],
        "s0 / s": [0, 0],
        "b * T": [1, -1],
        "T * v": [1, 0],
        "v * delta_v": [2, -2],
        "sqrt(alpha * b)": [1, -2],
        "a_lead / alpha": [0, 0],
        "v^delta/a_comfort": [1, -2],
        "delta_v^delta": [1, -1],
        "s^Gamma": [1, 0],
        "s_safe": [1, 0],
        "a_v_square": [1, -2],
        "factor_s": [0, 0],
        "factor_v_square": [0, 0],
        "factor_s_square": [0, 0],
    },
    "combination_prefix_expression": {
        "v / v0": ["div", "v", "v0"],
        "s0 / s": ["div", "s0", "s"],
        "b * T": ["mul", "b", "T"],
        "T * v": ["mul", "T", "v"],
        "v * delta_v": ["mul", "v", "delta_v"],
        "sqrt(alpha * b)": ["sqrt", "mul", "alpha", "b"],
        "a_lead / alpha": ["div", "a_lead", "alpha"],
        "v^delta/a_comfort": ["div", "pow", "v", "delta", "a_comfort"],
        "delta_v^delta": ["pow", "delta_v", "delta"],
        "s^Gamma": ["pow", "s", "Gamma"],
        "s_safe": [
            "add", "s_0", "add", "mul", "v", "T",
            "div", "mul", "v", "delta_v",
            "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",],
        "a_v_square": [
            "mul", "alpha",
            "n2", "div", "v", "v_0", ],
        "factor_s": [
            "div"
            "add", "s_0", "add", "mul", "v", "T",
            "div", "mul", "v", "delta_v",
            "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
            "s"],
        "factor_v_square": ["n2", "div", "v", "v_0"],
        "factor_s_square": ["n2", "div",
                            "add", "s_0", "add", "mul", "T", "v",
                            "div", "mul", "v", "delta_v",
                            "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                            "s"],
    }
}

LIBRARY_ARGS = {"tokens_args": DEFAULT_TOKEN_ARGS,
                "superparent_units": [1, -2],
                "superparent_names": ["IDM"],
                "superparent_prog": [
                    #                       ["sub", "alpha",
                    #                       "mul", "alpha",
                    #                       "add",
                    #                       "mul", "n2", "div", "v", "v0", "n2", "div", "v", "v0",
                    #                       "n2", "div",
                    #                       "add", "s0", "add", "mul", "T", "v",
                    #                       "div", "mul", "v", "delta_v",
                    #                       "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                    #                       "s"],    # 不行
                    ["sub", "alpha",
                     "mul", "alpha",
                     "add",
                     "n2", "n2", "div", "v", "v_0",
                     "n2", "div",
                     # "add", "s_0", "mul", "T", "v",
                     "add", "s_0", "add", "mul", "T", "v",
                     "div", "mul", "v", "delta_v",
                     "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                     "s"],  # all length:31 # 可以
                    #   ["div", "mul", "v", "delta_v", "s_0"]
                    # ["mul", "alpha",
                    #  "sub", "1",
                    #  "add",
                    #  "n2", "div", "v", "v_0",
                    #  "n2", "div",
                    #  "add", "s_0", "add", "mul", "T", "v",
                    #  "div", "mul", "v", "delta_v",
                    #  "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                    #  "s"
                    #  ],
                ]
                }

BOOL_ARGS = {
    # Data Setting
    "bool_use_true_trajectory": False,

    # DRL Setting
    "n_epochs": 20,
    "epochs_save_expression_number": 15,

    # early stop when convergence
    # "convergence_epoch_at_least": 3,  # 20 #7
    # "stop_loss_limit": 4.,  # 0.3 #4.

    # plot picture
    "bool_plot_intermediate_process": True,
    "plot_intermediate_process_limit": 0.95,  # plot following picture

    # DRL experience replay
    "bool_DRL_experience_replay": False,
    "experience_replay_rate": 0.02,
    "retain_data_rounds": 3,  # 2 3 5 7
    "exp_decay_epsilon_greedy_setting": {
        "decay_ratio": 0.99,
        "init_epsilon": 1.0,
        "min_epsilon": 0.5,
    },

    # DRL_PPO settings
    "n_workers": 1,

    # Calibration Setting
    "bool_use_rough_calibration": False,
    "calibration_conversion_limit": 0.6,

    # LLM Setting About Evaluation
    "bool_select_k_combinations": True,  # update(and evaluation), if need update, then start evaluation
    "n_selection_of_combinations": 8,
    "bool_compute_numerical_score": True,  # numerical_score
    "bool_compute_semantic_score": True,  # semantic_score

    # DRL or DRL+LLM  __parallel need, because increase number

    # single learning or incremental
    "bool_use_combinations": True,
    "bool_use_evolutions": True,
    "n_evolutions": 30,

    # LLM Setting About Extract
    "bool_extract_valuable": False,  # extract
    "bool_agent_single_step_increase": True,  # single increment by agent
    "new_tokens_number": 10,

    # LLM Setting About Reflection
    "bool_reflection": False,

    # LLM Setting About Explain
    "bool_explain_final_expressions": False,  # LLM explain

    # find old formula or new formula
    # reward_function=[ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY] or [DEFAULT_REWARD_FUNCTION],
    # only_built_directly=False or True,
    "bool_find_new_formula": False,

}

SAC_ARGS = {
    "actor_lr": 1e-3,  # 1e-3,0.0005
    "critic_lr": 1e-2,  # 1e-2,0.0007
    "alpha_lr": 1e-2,  # 1e-2,1e-3
    "target_entropy": -1,
    "tau": 0.01,
    "gamma": 0.99,  # discount
    "n_warmup_batches": 10,
    "sample_batch_size": DEFAULT_ENV_ARGS["batch_size"] * 12,  # 15
    "replay_buffer_capacity": 1000000,
}

AGENT_ARGS = {
    "fewshot_num": 3,
    "best_symbol_num": 3,  # 2 3_not_good # 3 for idm(only directly built)
    "delete_combination_num": 1,
    "best_expression_num": 5,
    "reflection_num": 1,
    "extend_num": 1,
    "max_try_num": 3,
}

DEFAULT_TRAIN_ARGS = {
    "env_args": DEFAULT_ENV_ARGS,
    "early_stop_args": DEFAULT_EARLY_STOP_ARGS,
    "token_args": DEFAULT_TOKEN_ARGS,
    "library_args": LIBRARY_ARGS,
    "model_type": DEFAULT_MODEL_TYPE,
    "model_args": DEFAULT_MODEL_ARGS,
    "sac_value_args": DEFAULT_SAC_VALUE_ARGS,
    "prior_args": DEFAULT_PRIOR_ARGS,
    "optimizer_args": DEFAULT_OPTIMIZER_ARGS,
    "bool_args": BOOL_ARGS,
    "sac_args": SAC_ARGS,
    "agent_args": AGENT_ARGS,
    "port": "7890",  # based on the port of the server #17890/15732
}
