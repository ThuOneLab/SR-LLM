from datetime import datetime
import os
import torch
import numpy as np
import random
import stopit

def get_r2_score_info_dict(idx, name, eq, infix_simplified_expre, r2, mae, prev):
    print("repeat_times index:",idx)
    print("dataset name:",name)
    print("origin equation:",eq)
    
    timeout=60
    with stopit.ThreadingTimeout(timeout) as tt:
        try:
            infix_str = str(infix_simplified_expre)
        except ZeroDivisionError:
            print("When easy_evaluate: sympy.sympify encountered ZeroDivisionError; returning unsimplified expression")
            infix_str = str(eq)+"(ZeroDivisionError)"
    if tt.state == tt.EXECUTED:
        print(f"simplify finished in {timeout}s")
    else:
        print(f"simplify timeout after {timeout}s, tt.state={tt.state}")
        infix_str = str(eq)+"(timeout)"
        
    print("final built expression:",infix_str)
    print("r2, mae:",r2, mae)
    print("need time:",datetime.now() - prev)
    print("\n")
    info_dict = {
        "idx": idx,
        "name": name,
        "eq": infix_str,
        "R ": r2,
        "mae": mae,
    }
    return info_dict

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def initialize_path(simple_initialize=False):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H-%M-%S")
    logPath = repo_root + "/log/"
    physo_logPath = repo_root + f"/physo_log/{formatted_time}"
    trainResultFolder = repo_root + "/trainResult/"
    md_folder = trainResultFolder + formatted_time + "/"
    if not simple_initialize and not os.path.exists(logPath):
        os.makedirs(logPath)
    if not simple_initialize and not os.path.exists(physo_logPath):
        os.makedirs(physo_logPath)
    if not simple_initialize and not os.path.exists(trainResultFolder):
        os.makedirs(trainResultFolder)
    if not simple_initialize and not os.path.exists(md_folder):
        os.makedirs(md_folder)
    return formatted_time, logPath, physo_logPath, trainResultFolder, md_folder

def initialize_train_args(train_args):
    '''batch_size'''
    batch_size = train_args["env_args"]["batch_size"]
    
    '''token_args'''
    token_args = train_args["token_args"]

    """initialize bool_args"""
    n_epochs = train_args["bool_args"]["n_epochs"]
    bool_use_evolutions = train_args["bool_args"]["bool_use_evolutions"]
    n_evolutions = train_args["bool_args"]["n_evolutions"] if bool_use_evolutions else 1
    have_got_semantic_score = False
    
    '''initialize agent_args'''
    k_best_of_symbols = train_args["agent_args"]["best_symbol_num"]
    k_delete_of_combinations = train_args["agent_args"]["delete_combination_num"]

    """early stop factor"""
    stop_reward = train_args["early_stop_args"]["stop_reward"]
    stop_after_n_epochs = train_args["early_stop_args"]["stop_after_n_epochs"]
    """loss setting"""
    gp_gamma_decay = train_args["env_args"]["gp_gamma_decay"]
    entropy_gamma_decay = train_args["env_args"]["entropy_gamma_decay"]
    entropy_weight = train_args["env_args"]["entropy_weight"]
    risk_factor = train_args["env_args"]["risk_factor"]
    n_keep = max(int(risk_factor * batch_size), 1)

    """parato setting"""
    fig_last_evolution = None
    ax_last_evolution = None

    """multiEnvs setting"""
    n_workers = train_args["bool_args"]["n_workers"]
    print("n_workers of multi_envs:", n_workers)
    
    return batch_size, token_args, n_epochs, bool_use_evolutions, n_evolutions, have_got_semantic_score, k_best_of_symbols, k_delete_of_combinations, stop_reward, stop_after_n_epochs, gp_gamma_decay, entropy_gamma_decay, entropy_weight, risk_factor, n_keep, fig_last_evolution, ax_last_evolution, n_workers

def breakout(visualiser, trainResult):
    visualiser.update_plot(trainResult)
    visualiser.save_visualization()
    # plt.show()