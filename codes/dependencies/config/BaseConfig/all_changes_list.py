from codes.dependencies.config.BaseConfig.baseConfig import *
from codes.dependencies.config.BaseConfig.baseConfig_update import *
'''
easy_benchmarks: 用于跑easy_benchmark codes\dependencies\config\FindExistingFormula\easy_benchmarks.py

FindExistingFormula中，不带有ngsim说明是人造数据，带有ngsim说明是ngsim真实数据

FindNewFormula中，不带有ngsim说明是人造数据，带有ngsim说明是ngsim真实数据

'''
'''1、easy_benchmarks:不使用evolutions'''
DEFAULT_EARLY_STOP_ARGS["stop_reward"] = 1.0 - 1e-5
DEFAULT_ENV_ARGS["max_time_step"] = 40 # 
DEFAULT_ENV_ARGS["reward_weight"] = {
        "factor_complexity": 0,
        "factor_similarity": 0,
        "factor_rmse": 1.0, # 只考虑r2
    }
update_LIBRARY_ARGS(LIBRARY_ARGS,formula='None')
BOOL_ARGS["bool_use_true_trajectory"]=False
BOOL_ARGS["bool_use_evolutions"]=False

'''2、ExsitingFormula 无ngsim'''
# 对于每个不同的公式，都分别有不同的Default_tokens_args和LIBRARY_ARGS里面的superparents
update_LIBRARY_ARGS(LIBRARY_ARGS,formula='IDM') # GHR Helly
# SAC_ARGS["n_warmup_batches"] = 10 #这个不要了
AGENT_ARGS["best_symbol_num"] = 3
AGENT_ARGS["delete_combination_num"] = 1
BOOL_ARGS["bool_use_true_trajectory"] = False

'''3、ExsitingFormula 有ngsim'''
# 对于每个不同的公式，都分别有不同的Default_tokens_args和LIBRARY_ARGS里面的superparents

'''4、NewFormula 有ngsim'''
update_LIBRARY_ARGS(LIBRARY_ARGS,formula='NEW')
BOOL_ARGS["bool_find_new_formula"] = True

'''最后通用的字典：每个都需要'''
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