import copy
import logging
from sklearn.metrics import r2_score
import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from codes.trafficSR.utils import *

from codes.trafficSR.A_sampling.SAC_utils.ReplayBuffer import ReplayBuffer
from codes.trafficSR.A_sampling.SAC_utils.SAC import SAC
from codes.trafficSR.A_sampling.env_generator import SRenv
from codes.trafficSR.A_sampling import SRenvs
from codes.trafficSR.A_sampling.policy_network.policy_network import PolicyNetwork
from codes.trafficSR.A_sampling.prefix_tree.PrefixTree import CompressedPrefixTree

from codes.trafficSR.C_evaluation.SRUCB_for_expand_combination import (
    UCB_for_expand_combination,
)
from codes.trafficSR.C_evaluation.select_best_worst import LLM_select_best_worst_for_rag

from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT

from codes.trafficSR.E_train_codes.train_utils.RLtrainUtils import (
    initialize_path,
    initialize_train_args,
    breakout,
)
from codes.trafficSR.E_train_codes.train_utils.RecordUtils import (
    envInitLogging,
    epoch_log_info,
    evolution_log_info,
    write_evolution_different_program_md,
)
from codes.trafficSR.E_train_codes.train_utils.EarlyStopUtils import early_stop
from codes.trafficSR.E_train_codes.train_monitor.train_result_record import ResultRecord
from codes.trafficSR.E_train_codes.train_monitor.run_logger import RunLogger
from codes.trafficSR.E_train_codes.train_loss_optimizer.SRloss import loss_func
from codes.trafficSR.E_train_codes.train_loss_optimizer.SRoptimizer import get_optimizer
from codes.trafficSR.E_train_codes.train_monitor.SRvisualization import (
    Visualiser,
    print_c,
    simplify_True_length_limit,
)


# torch.autograd.set_detect_anomaly(True)
evolution_colors = ["b", "g", "r"]


def RL_LLM_train(
    X,
    y,
    train_args=None,
    ngsim_args=None,
    data_source="NGSIM_multiVehicle/MultiVehicle4",
    seed=100,
    memory_path="codes/ragLibrary/memory_idm",
):
    """
    X: (data_size, input_dim)
    y: (data_size, )
    """
    """initialize path"""
    formatted_time, logPath, physo_logPath, trainResultFolder, md_folder = (
        initialize_path()
    )

    """initialize agent and recorder"""
    agent = RAG_AGENT(
        trainResultFolder=md_folder,
        port=train_args["port"],
        address=train_args["address"],
        fewshot_num=train_args["agent_args"]["fewshot_num"],
        extend_num=train_args["agent_args"]["extend_num"],
        reflection_num=train_args["agent_args"]["reflection_num"],
        time=formatted_time,
        memory_path=memory_path,
        only_directly_built=False if train_args["bool_args"]["bool_find_new_formula"] else True, # 是否希望直接构建
        bool_is_feynman=train_args["bool_args"]["bool_is_feynman"],
    )
    agent.knowledge_pool.read_target_names()
    run_logger = RunLogger(save_folder=physo_logPath, do_save=True)
    result_record = ResultRecord(
        bool_args=train_args["bool_args"], ngsim_args=ngsim_args
    )
    visualiser = Visualiser(
        save_path="pictures/A_sampling/weighted_score_density/" + formatted_time + "/"
    )

    """initialize args"""
    device = train_args["env_args"]["device"]
    dtype = train_args["env_args"]["dtype"]
    if "x_lead" in ngsim_args.keys():
        ngsim_args["x_lead"] = ngsim_args["x_lead"].to(device, dtype)
        ngsim_args["v_lead"] = ngsim_args["v_lead"].to(device, dtype)
        ngsim_args["x_obs"] = ngsim_args["x_obs"].to(device, dtype)
        ngsim_args["v_obs"] = ngsim_args["v_obs"].to(device, dtype)
    X_numpy = X.detach().numpy()
    y_numpy = y.detach().numpy()
    
    '''train_args'''
    batch_size, token_args, n_epochs, bool_use_evolutions, n_evolutions, have_got_semantic_score, k_best_of_symbols, k_delete_of_combinations, stop_reward, stop_after_n_epochs, gp_gamma_decay, entropy_gamma_decay, entropy_weight, risk_factor, n_keep, fig_last_evolution, ax_last_evolution, n_workers=initialize_train_args(train_args)
    print("max_time_step: ", train_args["env_args"]["max_time_step"])
    """PrefixTree"""
    prefix_tree = CompressedPrefixTree()

    """UCB for score calculate, so exploring symbols that are rarely expanded"""
    combination_UCB = UCB_for_expand_combination(k_best_of_symbols=k_best_of_symbols)

    """directly built"""
    directly_built = []
    R_last_epoch = []
    
    '''epoch渐渐升高,因为构建combination后，后面的epoch更容易达到最优'''
    bool_epoch_increase = train_args["bool_args"]["bool_is_feynman"] and train_args["bool_args"]["bool_use_evolutions"]
    if bool_epoch_increase:
        max_n_epochs=60
        increase_step=int((max_n_epochs-n_epochs)/(n_evolutions-1)) # n_epoch会逐渐增加，最终在evolution9时变成接近n_epoch*k.

    for evolution in range(n_evolutions):
        result_record.reset() # reset result_record!
        """init envs"""
        envs = SRenvs.MultiprocessEnv(
            X_numpy=X_numpy,
            y_numpy=y_numpy,
            train_args=train_args,
            token_args=token_args,
            ngsim_args=ngsim_args,
            prefix_tree=prefix_tree,
            batch_size=batch_size,
            seed=seed,
            n_workers=n_workers,
        )
        """use envs to initialize"""
        envInitLogging(
            logging,
            logPath,
            formatted_time,
            data_source,
            evolution,
            envs[0],
            train_args,
        )
        prefix_tree.reset_names_lib(envs[0])
        prefix_tree.reset_N_table()
        policy_network = PolicyNetwork(train_args, envs[0].tokens_number, device, dtype)
        optimizer = get_optimizer(
            model=policy_network.model, **train_args["optimizer_args"]
        )
        replay_buffer = ReplayBuffer(
            capacity=train_args["sac_args"]["replay_buffer_capacity"], rank_based=True # True
        )
        sac_agent = SAC(
            policy_network,
            train_args,
            envs[0].tokens_number,
            device,
            dtype,
            envs,
            prior_args=train_args["prior_args"],
        )
        reinforce_loss_numpy = 100  # means not in SAC mode
        """initialize UCB info"""
        combination_UCB.init_UCB_info(
            envs.main_env
        )  # init UCB for combination that first appear in library
        overall_best_r2_of_evolution = 0
        """epoch iteration"""
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            exploration = True
            """get trajectory"""
            (
                logits,
                actions_numpy,
                observations_overall,
                observations_partial,
                R,
                R_SUB,
                R_similarity,
                states_overall,
                states_partial,
                prior_UCBs,
            ) = envs.getTraj(train_args, policy_network, prefix_tree, epoch)
            # 判断logits、R中是否有nan

            assert not torch.isnan(logits).any(), "logits has nan!"
            assert not np.isnan(R).any(), "R has nan!"

            if epoch == n_epochs - 1:
                R_last_epoch.append(np.array(R))
            states_overall = states_overall.detach()
            states_partial = states_partial.detach()
            # setup_seed(seed)
            y = torch.tensor(y_numpy).to(device, dtype)

            """update prefix_tree and tokens' scores"""
            for env_id in range(len(envs)):
                prefix_tree.update_batch_per_epoch(envs[env_id])
            prefix_tree.update_tokens_score(print_score=False)

            """get keep"""
            keep = R.argsort()[::-1][0:n_keep].copy()  # (n_keep,), large to small
            notkept = R.argsort()[::-1][n_keep:].copy()  # (batch_size-n_keep,), large to small
            """get candidates and reward"""
            R_train = R[keep] # R是numpy
            R_train_sub = R_SUB[keep]

            """更新result_record的epoch reward和different_expressions"""
            """ update reward and elites in result_record"""
            best_prog_id = R.argmax()
            result_record.update_epoch_reward_and_elites(
                R, R_SUB, best_prog_id, envs, R_train, R_train_sub
            )

            """update different expressions in result_record"""
            # TODO：这里的div sin 1 log 1不太对，以后费尔曼再打开
            result_record.update_epoch_different_expressions(
                R_train,
                keep,
                R,
                R_SUB,
                envs,
                update_number=train_args["bool_args"]["epochs_save_expression_number"]* 2,
                need_simply=False,
            )
            """update action and prob first"""
            # Elite candidates logprobs
            logits_train = logits[:, keep]  # (max_time_step, n_keep, tokens_number,)
            # Elite candidates as one-hot target probs——tokens_number is number of all available tokens
            actions_array_train = actions_numpy[:, keep]  # (max_time_step, n_keep,)
            ideal_probs_train = np.eye(envs[0].tokens_number)[
                actions_array_train
            ]  # (max_time_step, n_keep, tokens_number,)
            ideal_probs_train = torch.tensor(
                ideal_probs_train.astype(np.float32),
                requires_grad=False,
            ).to(device)
            lengths = envs.main_env.programs.valid_lengths[keep]  # (n_keep,) Lengths of programs

            """add great elites to buffer"""
            n_physical = np.sum(envs.main_env.library.is_physical) #发现问题：有可能没有completed，但是physical?——不会
            n_physical=min(n_physical, n_keep)
            if n_physical > 0:
                add_prog_id = R.argsort()[::-1][0:n_physical].copy()
                add_prog_id_list = add_prog_id.tolist()
                observations_overall_add = observations_overall[
                    :, add_prog_id_list
                ].transpose(1, 0, 2)
                observations_partial_add = observations_partial[
                    :, add_prog_id_list
                ].transpose(1, 0, 2)
                states_overall_add = states_overall[
                    :, :, :, add_prog_id_list, :
                ].transpose(3, 0)
                states_partial_add = states_partial[
                    :, :, :, add_prog_id_list, :
                ].transpose(3, 0)
                actions_add = actions_numpy[
                    :, add_prog_id
                ].T  # (max_time_step, n_add) for loss calculate
                rewards_add = R[add_prog_id]
                prior_UCBs_add = prior_UCBs[:, add_prog_id_list].transpose(1, 0)
                replay_buffer.add_trajectory(
                    obsers_overall=observations_overall_add,
                    obsers_partial=observations_partial_add,
                    actions=actions_add,
                    rewards=rewards_add,
                    valid_lengths=envs.main_env.programs.valid_lengths[add_prog_id],
                    states_overall=states_overall_add,
                    states_partial=states_partial_add,
                    prior_UCBs=prior_UCBs_add,
                )

            """add units_unconsistency to buffer"""
            units_inconsistency_index = np.where(
                envs.main_env.library.units_inconsistency > 0
            )[0]  # only add unphysical programs，对其进行惩罚
            units_inconsistency_index = units_inconsistency_index[
                :min(int(0.6*n_keep), min(n_physical, units_inconsistency_index.shape[0]))
            ] # 最多添加m*n_keep个
            if units_inconsistency_index.shape[0] > 0:
                add_prog_id_list = units_inconsistency_index.tolist()
                observations_overall_add = observations_overall[
                    :, add_prog_id_list
                ].transpose(1, 0, 2)
                observations_partial_add = observations_partial[
                    :, add_prog_id_list
                ].transpose(1, 0, 2)
                states_overall_add = states_overall[
                    :, :, :, add_prog_id_list, :
                ].transpose(3, 0)
                states_partial_add = states_partial[
                    :, :, :, add_prog_id_list, :
                ].transpose(3, 0)
                actions_add = actions_numpy[
                    :, units_inconsistency_index
                ].T  # (max_time_step, n_add) for loss calculate
                rewards_add = [-0.2] * units_inconsistency_index.shape[0]
                prior_UCBs_add = prior_UCBs[:, add_prog_id_list].transpose(1, 0)
                replay_buffer.add_trajectory(
                    obsers_overall=observations_overall_add,
                    obsers_partial=observations_partial_add,
                    actions=actions_add,
                    rewards=rewards_add,
                    valid_lengths=envs.main_env.library.units_inconsistency[
                        units_inconsistency_index
                    ], #是一直选到那个单位不一致的位置
                    states_overall=states_overall_add,
                    states_partial=states_partial_add,
                    prior_UCBs=prior_UCBs_add,
                )

            """SAC Sampling or Reinforcement Algorithm"""
            if replay_buffer.n_experiences > sac_agent.min_samples and reinforce_loss_numpy < 0.05:
                print_c("----------------------SAC Algorithm",color=33)
                sac_agent.update_abs_td_error(replay_buffer)# update abs_td_error，用于经验池优先级采样PER
                transition_dict = replay_buffer.sample_token_idx_from_buffer(
                    batch_size=min(replay_buffer.len(), sac_agent.sample_batch_size)
                )
                sac_actor_loss_numpy = sac_agent.update(transition_dict) # 返回的是策略网络actor_loss_numpy
                loss_val_numpy = sac_actor_loss_numpy
            else:
                print_c("----------------------Reinforcement Algorithm",color=33)
                baseline = R_train.min()
                
                loss_val = loss_func(
                    logits_train=logits_train,
                    ideal_probs_train=ideal_probs_train,
                    R_train=torch.tensor(R_train).to(device), # requeries_grad=False
                    baseline=torch.tensor(baseline).to(device),
                    lengths=lengths,
                    gp_gamma_decay=gp_gamma_decay,
                    entropy_gamma_decay=entropy_gamma_decay,
                    entropy_weight=entropy_weight,
                    device=device,
                )
                result_record.trainResult["loss_history"].append(
                    loss_val.detach().cpu().numpy()
                )
                loss_val.backward()
                optimizer.step()
                reinforce_loss_numpy = loss_val.detach().cpu().numpy()
                loss_val_numpy = reinforce_loss_numpy
                
                '''在reinforce期间，就已经把critic和alpha进行更新。但是actor由reinforce更新'''
                if R_train.max() > 0.:
                    print_c("----------------------SAC Algorithm (no actor update)",color=33)
                    sac_agent.update_abs_td_error(replay_buffer)# update abs_td_error，用于经验池优先级采样PER
                    transition_dict = replay_buffer.sample_token_idx_from_buffer(
                        batch_size=min(replay_buffer.len(), sac_agent.sample_batch_size)
                    )
                    sac_agent.update(transition_dict, bool_actor_update=False) # 只更新Q网络和alpha
            
            """更新result_record的loss_history和best_program"""
            """get loss and backward"""
            result_record.trainResult["loss_history"].append(loss_val_numpy)

            """update best program and free const values"""
            result_record.update_evolution_best_program(R, R_SUB, envs, y, X_numpy, device)
            result_record.update_evolution_similarity_best_sub_reward(R_SUB)
            run_logger.log(
                epoch=epoch,
                batch=envs.main_env,
                model=policy_network.model,
                rewards=R,
                rewards_sub=R_SUB,
                keep=keep,
                notkept=notkept,
                loss_val=loss_val_numpy,
            )
            # fig, ax = plot_pareto_front(run_logger, fig=fig_last_evolution, ax=ax_last_evolution,
            #                             color=evolution_colors[evolution])

            """这部分是epoch信息更新"""
            """epoch log info"""
            r2_of_epoch=r2_score(y_numpy.reshape(-1), result_record.trainResult["candidates_history"][-1](X).detach().cpu().numpy())
            overall_best_r2_of_evolution = max(overall_best_r2_of_evolution, r2_of_epoch)
            epoch_log_info(
                logging,
                evolution,
                epoch,
                R,
                R_SUB,
                best_prog_id,
                envs,
                loss_val_numpy,
                R_train,
                result_record,
                buffer_from_exploration_length=0 if exploration else n_keep,
                r2=r2_of_epoch,
                overall_best_r2=overall_best_r2_of_evolution,
            )

            """epoch visualization"""
            visualiser.visualise(epoch, result_record.trainResult)

            """epoches result_record.trainResult record for plot"""
            agent.record_trainResult_history(
                trainResult=result_record.trainResult,
                writePath=md_folder + "reward_history_evolution" + f"{evolution}.md",
            )  # for plot_reward_history

            """Judge Using Rough calibration and Fine calibration"""
            if (
                envs[0].bool_use_rough_calibration
                and result_record.trainResult["mean_R_train_history"][-1]
                >= train_args["bool_args"]["calibration_conversion_limit"]
            ):
                envs.false_bool_use_rough_calibration()

            """save different program and free const values"""
            write_evolution_different_program_md(
                trainResult=result_record.trainResult,
                agent=agent,
                md_folder=md_folder,
                evolution=evolution,
                save_number=train_args["bool_args"]["n_epochs"]
                * train_args["bool_args"]["epochs_save_expression_number"],
            )

            '''early stop for epochs'''
            stop_after_n_epochs,bool_early_stop=early_stop(result_record, stop_reward, stop_after_n_epochs, R_similarity, logging, visualiser)
            if bool_early_stop:
                breakout(visualiser, result_record.trainResult)
                break

        """parato_plot
        # fig.savefig(os.path.join(run_logger.save_folder, f"evolution{evolution}.jpg"))
        # plt.close(fig)
        # fig_last_evolution = fig
        # ax_last_evolution = ax
        """
        """evolution信息更新"""
        evolution_log_info(
            logging,
            result_record.trainResult,
            result_record.best_free_const_values,
            result_record.best_a_loss,
        )

        """save evolution best N program and free const values"""
        n_save_best = min(
            len(result_record.trainResult["different_rewards_history"]), 5
        )
        reward_sort = np.argsort(
            np.array(result_record.trainResult["different_rewards_history"])
        )[::-1]
        for i in range(n_save_best):
            best_prog_id = reward_sort[i]
            result_record.evolutionBest_append_trainResult(best_prog_id)
        
        '''early stop for evolutions'''
        if bool_early_stop:
            break

        """Select the best k combinations to avoid excessive search space"""
        if bool_use_evolutions:
            """initialize env"""
            # 使用的是最后一个evolution的different_expressions
            code_blocks = agent.extract_python_code_blocks(
                f"{agent.trainResultFolder}different_expressions_evolution{evolution}.md"
            )
            different_expression = eval(code_blocks[-1])
            reward_of_expressions = [
                v["sum_reward"] for (k, v) in different_expression.items()
            ]

            # change super_parent anchor in library_args
            def change_super_parent_anchor(train_args_copy, different_expression):
                train_args_copy["library_args"]["superparent_prog"] = []
                for i, (k, v) in enumerate(different_expression.items()):
                    train_args_copy["library_args"]["superparent_prog"].append(
                        list(eval(v["prefix_expression"][1:-1]))
                    )  # string as a whole to list
                return train_args_copy

            train_args_copy = copy.deepcopy(train_args)
            train_args_copy = change_super_parent_anchor(
                train_args_copy=train_args_copy,
                different_expression=different_expression,
            )
            now_token_args = copy.deepcopy(token_args)
            new_env = SRenv.SRenv(  # get new env(from now_token_args)
                X_numpy=X_numpy,
                y_numpy=y_numpy,
                library_args=train_args_copy["library_args"],
                env_args=train_args_copy["env_args"],
                token_args=now_token_args,
                ngsim_args=ngsim_args,
                bool_args=train_args_copy["bool_args"],
            )
            """select best for updated new_combinations"""
            (
                best_symbols,
                worst_combinations,
                best_expressions,
                have_got_semantic_score,
            ) = LLM_select_best_worst_for_rag(
                agent,
                new_env,
                different_expression,
                reward_of_expressions,
                evolution,
                have_got_semantic_score,
                k_best_of_symbols,
                k_delete_of_combinations,
                k_best_of_expressions=train_args["agent_args"]["best_expression_num"],
                combination_UCB=combination_UCB,
                directly_built=directly_built,
            )

           

            """update UCB for combination"""
            combination_UCB.update_UCB_info(best_symbols, worst_combinations, new_env)

            """ Combine new symbols by LLM """
            for i, (k, v) in enumerate(best_symbols.items()):
                from codes.dependencies.config.FindNewFormula.IDM import DEFAULT_TOKEN_ARGS as IDM_COMBINATIONS
                new_symbol_name = IDM_COMBINATIONS["combination_tokens"][evolution]
                for _ in range(1):
                    # generate new symbol with LLM
                    new_symbol_dictionarys = [
                        {
                            "name":new_symbol_name,
                            "description":IDM_COMBINATIONS["combination_description"][new_symbol_name],
                            "prefix_expression":new_symbol_name,
                            "type":"combination",
                            "units":IDM_COMBINATIONS["combination_units"][new_symbol_name],
                        }
                    ]
                    # update token_args
                    token_args = agent.update_token_args_combinations_RAG(
                        token_args, new_symbol_dictionarys
                    )
            
                    
    breakout(visualiser, result_record.trainResult)
    agent.write_all_evolutions_best_expressions(
        bestInEvolutions=result_record.bestInEvolutions,
        writePath=f"{agent.trainResultFolder}best_result_of_evolutions.md",
    )
    
    if train_args["bool_args"]["bool_explain_final_expressions"]:
        agent.explain_expressions(
            origin_token_dict=token_args, bestResultFile="best_result_of_evolutions.md"
        )
    
    if len(result_record.bestInEvolutions["best_R_history"]) == 0:
        return None, None, None
    else:
        '''获取self.bestInEvolutions["best_R_history"]当中的最大值对应的index'''
        best_prog_id = result_record.bestInEvolutions["best_R_history"].index(
            max(result_record.bestInEvolutions["best_R_history"])
        )
        '''获取各轮evolutions中最好的表达式'''
        best_program_in_evolutions = result_record.bestInEvolutions["best_program_history"][best_prog_id]
        '''返回相关值'''
        return (
            np.array(R_last_epoch),
            best_program_in_evolutions.get_infix_sympy(do_simplify=True, no_const_subtitution=False),
            best_program_in_evolutions(X),
        )
