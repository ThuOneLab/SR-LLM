import numpy as np
import copy
from codes.dependencies.config.BaseConfig.baseConfig import *

'''EARLY_STOP_ARGS'''
def update_EARLY_STOP_ARGS(BASE_EARLY_STOP_ARGS,formula='IDM'):
    # if formula in ['EASY', 'FEYN']: 
    #     BASE_EARLY_STOP_ARGS["stop_reward"] = 1.0 - 1e-5
    return BASE_EARLY_STOP_ARGS

'''ENV_ARGS'''
def update_ENV_ARGS(BASE_ENV_ARGS,formula='IDM'):
    # BASE_ENV_ARGS["max_time_step"]原本是45
    if formula == 'IDM_artifical':
        BASE_ENV_ARGS["max_time_step"] = 60
        BASE_ENV_ARGS["reward_weight"] = {
            "factor_complexity": 0,
            "factor_similarity": 0.2,
            "factor_rmse": 0.8,
        }
    elif formula == 'GHR_artifical' or formula == 'Helly_artifical':
        BASE_ENV_ARGS["max_time_step"] = 20
        BASE_ENV_ARGS["reward_weight"] = {
            "factor_complexity": 0,
            "factor_similarity": 0.2,
            "factor_rmse": 0.8,
        }
    return BASE_ENV_ARGS

'''PRIOR_ARGS'''
def update_PRIOR_ARGS(BASE_PRIOR_ARGS,formula='IDM'):
    if formula == 'GHR' or formula == 'GHR_artifical':
        BASE_PRIOR_ARGS["prior_config"]["SoftLengthPrior"]={"length_loc": 7, "scale": 5, "eps": 1e-2}
    elif formula == 'Helly' or formula == 'Helly_artifical':
        if formula == 'Helly' and 'SoftMaxLength' in BASE_PRIOR_ARGS["prior_type"]:
            BASE_PRIOR_ARGS["prior_type"].remove('SoftMaxLength')
            if 'SoftLength' not in BASE_PRIOR_ARGS["prior_type"]:
                BASE_PRIOR_ARGS["prior_type"].append('SoftLength') # Helly: 将其SoftMaxLength改为SoftLength
        BASE_PRIOR_ARGS["prior_config"]["SoftLengthPrior"]={"length_loc": 13, "scale": 9, "eps": 1e-2}
    return BASE_PRIOR_ARGS

'''TOKEN'''
def update_TOKEN_ARGS(BASE_TOKEN_ARGS,formula='IDM'):
    if formula == 'IDM' or formula == 'IDM_artifical':
        BASE_TOKEN_ARGS["free_const_tokens"] = ["alpha", "b", "v_0", "T", "s_0"] #之前IDM有"c", "k_1", "k_2"
    elif formula == 'GHR' or formula == 'GHR_artifical':
        BASE_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"] #之前GHR有"n2" "sqrt"
        BASE_TOKEN_ARGS["variable_tokens"] = ["delta_x", "v", "delta_vi"] #注意这里的s变成了delta_x，delta_v变成了delta_vi
        BASE_TOKEN_ARGS["free_const_tokens"] = ["c"]
    elif formula == 'Helly' or formula == 'Helly_artifical':
        BASE_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"] #之前Helly有"n2" "sqrt"
        BASE_TOKEN_ARGS["variable_tokens"] = ["delta_x", "v", "delta_vi"]
        BASE_TOKEN_ARGS["free_const_tokens"] = ["Beta", "k_1", "k_2"]
        BASE_TOKEN_ARGS["fixed_const_tokens"] = ["1","s_20"]
    elif formula == 'NEW':
        pass

    return BASE_TOKEN_ARGS

'''LIBRARY'''
default_superparents=[
    # ngsim版本
                    ["sub", "alpha",
                     "mul", "alpha",
                     "add",
                     "n2", "n2", "div", "v", "v_0",
                     "n2", "div",
                     "add", "s_0", "add", "mul", "T", "v",
                     "div", "mul", "v", "delta_v",
                     "add", "sqrt", "mul", "alpha", "b", "sqrt", "mul", "alpha", "b",
                     "s"
                     ],  # idm
                    ["mul",
                     "c",
                     "div",
                     "mul",
                     "v", "delta_v",
                     "s",
                     ],  # ghr
                    ["add",
                     "mul", "k_1", "delta_v",
                     "mul", "k_2",
                     "sub", "s", "s_0"
                     ],  # helly
                ]

def update_LIBRARY_ARGS(BASE_LIBRARY_ARGS,TOKEN_ARGS,formula='IDM', pb=None):
    if formula == 'IDM' or formula == 'IDM_artifical':
        BASE_LIBRARY_ARGS["superparent_prog"]=[default_superparents[0]]
    elif formula == 'GHR' or formula == 'GHR_artifical':
        BASE_LIBRARY_ARGS["superparent_prog"]=[default_superparents[3]]
    elif formula == 'Helly' or formula == 'Helly_artifical':
        BASE_LIBRARY_ARGS["superparent_prog"]=[default_superparents[4]]
    elif formula == 'NEW':
        BASE_LIBRARY_ARGS["superparent_prog"]=default_superparents[:3] # ngsim版本
        
    BASE_LIBRARY_ARGS["tokens_args"]=TOKEN_ARGS #因为之前可能Token_args被修改了，所以这里要重新赋值
    return BASE_LIBRARY_ARGS

'''Agent'''
def update_Agent_ARGS(BASE_AGENT_ARGS,formula='IDM'):
    if formula == 'IDM' or formula == 'IDM_artifical':
        BASE_AGENT_ARGS["delete_combination_num"] = 1
        if formula == 'IDM':
            BASE_AGENT_ARGS["best_symbol_num"] = 3
    elif formula == 'GHR' or formula == 'GHR_artifical':
        BASE_AGENT_ARGS["delete_combination_num"] = 1
    elif formula == 'Helly' or formula == 'Helly_artifical':
        BASE_AGENT_ARGS["fewshot_num"] = 2
        BASE_AGENT_ARGS["delete_combination_num"] = 1
    elif formula == 'NEW':
        pass
    return BASE_AGENT_ARGS

'''BOOL_ARGS'''
def update_BOOL_ARGS(BASE_BOOL_ARGS,formula='IDM'):
    # BASE_BOOL_ARGS["bool_use_true_trajectory"]本身是True
    # BASE_BOOL_ARGS["n_epochs"]本身是20
    # BASE_BOOL_ARGS["n_evolutions"]本身是30
    # BASE_BOOL_ARGS["bool_use_evolutions"]本身是True
    # BASE_BOOL_ARGS["bool_find_new_formula"]本身是False
    if formula == 'IDM_artifical':
        BASE_BOOL_ARGS["bool_find_new_formula"] = True
        BASE_BOOL_ARGS["n_epochs"]=40
        BASE_BOOL_ARGS["n_evolutions"]=15
        BASE_BOOL_ARGS["bool_use_evolutions"]=True
    elif formula == 'IDM':
        pass
    elif formula == 'GHR_artifical':
        BASE_BOOL_ARGS["bool_use_true_trajectory"]=False
        BASE_BOOL_ARGS["n_epochs"]=10
        BASE_BOOL_ARGS["n_evolutions"]=20
    elif formula == 'GHR':
        BASE_BOOL_ARGS["n_epochs"]=10
        BASE_BOOL_ARGS["n_evolutions"]=20
        # BASE_BOOL_ARGS["n_epochs"]=200
        # BASE_BOOL_ARGS["bool_use_evolutions"]=False
    elif formula == 'Helly_artifical':
        BASE_BOOL_ARGS["bool_use_true_trajectory"]=False
        BASE_BOOL_ARGS["n_epochs"]=7
        BASE_BOOL_ARGS["n_evolutions"]=20
    elif formula == 'Helly':
        BASE_BOOL_ARGS["n_epochs"]=7
        BASE_BOOL_ARGS["n_evolutions"]=20
    elif formula == 'NEW':
        BASE_BOOL_ARGS["bool_find_new_formula"] = True
        BASE_BOOL_ARGS["n_epochs"]=20
        BASE_BOOL_ARGS["n_evolutions"]=15
        BASE_BOOL_ARGS["bool_use_evolutions"]=True
        # 额外：drl的时候，BASE_BOOL_ARGS["n_epochs"]=50。并且["bool_use_evolutions"]=False
        
    return BASE_BOOL_ARGS

'''Entire'''
def get_DEFAULT_TRAIN_ARGS(
    BASE_MODEL_TYPE=DEFAULT_MODEL_TYPE, #由于字符串的不可变性，它直接生成一个新字符串（执行任何看起来像是“修改”字符串的操作，实际上会创建一个新的字符串对象）
    BASE_MODEL_ARGS=DEFAULT_MODEL_ARGS, # 对于字典：DEFAULT_MODEL_ARGS.copy() is DEFAULT_MODEL_ARGS False; DEFAULT_MODEL_ARGS.copy() == DEFAULT_MODEL_ARGS True。但是发现：浅拷贝会复制字典本身，但对于字典中包含的复杂对象（如列表、另一个字典等）只复制引用。如果字典中包含列表、集合或其他字典等可变对象，并且你需要完全独立于原字典的一个副本，则应使用深拷贝。
    BASE_OPTIMIZER_ARGS=DEFAULT_OPTIMIZER_ARGS,
    BASE_EARLY_STOP_ARGS=DEFAULT_EARLY_STOP_ARGS,
    BASE_ENV_ARGS=DEFAULT_ENV_ARGS,
    BASE_PRIOR_ARGS=DEFAULT_PRIOR_ARGS,
    BASE_TOKEN_ARGS=DEFAULT_TOKEN_ARGS,
    BASE_LIBRARY_ARGS=DEFAULT_LIBRARY_ARGS,
    BASE_AGENT_ARGS=DEFAULT_AGENT_ARGS,
    BASE_SAC_VALUE_ARGS=DEFAULT_SAC_VALUE_ARGS,
    BASE_SAC_ARGS=DEFAULT_SAC_ARGS,
    BASE_BOOL_ARGS=DEFAULT_BOOL_ARGS,
    formula="IDM", #formula带有artifical说明是人造数据，否则是ngsim真实数据
    pb=None
):
    assert formula in ["IDM", "GHR", "Helly", "IDM_artifical", "GHR_artifical", "Helly_artifical", "NEW", "EASY", "FEYN", "RANDOM"], \
        "formula must be in ['IDM', 'GHR', 'Helly', 'IDM_artifical', 'GHR_artifical', 'Helly_artifical', 'NEW', 'EASY', 'FEYN', 'RANDOM']"
    assert not(pb and formula != "FEYN"), "only FEYN formula can use pb"  
    EARLY_STOP_ARGS = update_EARLY_STOP_ARGS(copy.deepcopy(BASE_EARLY_STOP_ARGS), formula)
    ENV_ARGS = update_ENV_ARGS(copy.deepcopy(BASE_ENV_ARGS), formula)
    
    PRIOR_ARGS=update_PRIOR_ARGS(copy.deepcopy(BASE_PRIOR_ARGS), formula)
    
    TOKEN_ARGS = update_TOKEN_ARGS(copy.deepcopy(BASE_TOKEN_ARGS), formula)
    LIBRARY_ARGS = update_LIBRARY_ARGS(
        copy.deepcopy(BASE_LIBRARY_ARGS), TOKEN_ARGS, formula
    )  # 需要传入TOKEN_ARGS
    
    AGENT_ARGS = update_Agent_ARGS(copy.deepcopy(BASE_AGENT_ARGS), formula)
    
    BOOL_ARGS = update_BOOL_ARGS(copy.deepcopy(BASE_BOOL_ARGS), formula)
    
    DEFAULT_TRAIN_ARGS = {
        "model_type": copy.deepcopy(BASE_MODEL_TYPE),
        "model_args": copy.deepcopy(BASE_MODEL_ARGS),
        
        "optimizer_args": copy.deepcopy(BASE_OPTIMIZER_ARGS),
        "early_stop_args": EARLY_STOP_ARGS,
        
        "env_args": ENV_ARGS,
        "prior_args": PRIOR_ARGS,
        "token_args": TOKEN_ARGS,
        "library_args": LIBRARY_ARGS,
        
        "sac_value_args": copy.deepcopy(BASE_SAC_VALUE_ARGS),
        "sac_args": copy.deepcopy(BASE_SAC_ARGS),
        
        "agent_args": AGENT_ARGS,
        
        "bool_args": BOOL_ARGS,
        "address": "172.22.0.1",  # wsl: 172.22.0.1
        
        "port": "7890",  # based on the port of the server #7890/15732/wsl:7890
    }
    return copy.deepcopy(DEFAULT_TRAIN_ARGS)