import copy
import torch
from codes.trafficSR.B_calibration.SRcalibration_func import MSE_loss
from codes.trafficSR.E_train_codes.train_monitor.SRvisualization import simplify_True_length_limit


class ResultRecord():
    def __init__(self, bool_args, ngsim_args):
        self.bool_args = bool_args
        self.ngsim_args = ngsim_args
        self.trainResult = {
            # each epoch's candidates and best reward
            "sub_rewards_history": [], "rewards_history": [], "candidates_history": [], "free_const_history": [], "semi_free_const_history": [],
            # each epoch's R_train
            "R_train_history": [], "mean_R_train_history": [],
            # each epoch's mean of all rewards
            "mean_R_history": [],
            # mean epoch's sub reward of R_train
            "mean_R_train_sub_history": [],
            # each epoch's loss
            "loss_history": [],
            # evolution best reward history and overall best expression in one evolution(根据best_R来更新)
            "best_R_history": [], "best_sub_R_history": [], "best_program": None,
            # evolution best sub reward history(根据similarity来更新，从而记录最相似的样本，目前可以弃用)
            "similarity_best_R_sub_history": [],
        
            # for different expressions in one evolution
            "different_sub_rewards_history": [], "different_rewards_history": [],
            "different_candidates_history": [], "different_candidates_infix_history": [],
            "different_candidates_tokens_idx": [], "different_free_const_values_history": [],
            "different_semi_free_const_values_history": [],
        }
        self.bestInEvolutions = {"best_R_history": [], "best_sub_R_history": [], "best_program_history": [], "best_free_const_values_history": [],  "best_semi_free_const_values_history": []}
        self.best_free_const_values = None
        self.best_semi_free_const_values = None
        self.best_a_loss = 0
        self.best_v_loss = 0
        self.best_x_loss = 0
        self.best_prog_id = 0

    def reset(self):
        self.best_free_const_values = None
        self.best_semi_free_const_values = None
        self.best_a_loss = 0
        self.best_v_loss = 0
        self.best_x_loss = 0
        self.best_prog_id = 0
        self.reset_trainResult()

    def reset_trainResult(self):
        for key in self.trainResult:
            if type(self.trainResult[key]) == list:
                self.trainResult[key] = []
            elif type(self.trainResult[key]) == dict:
                self.trainResult[key] = {}
            else:
                self.trainResult[key] = None

    def update_epoch_reward_and_elites(self, R, R_SUB, best_prog_id, envs, R_train, R_train_sub):
        self.best_prog_id = best_prog_id
        self.trainResult["rewards_history"].append(R[best_prog_id])
        self.trainResult["sub_rewards_history"].append(R_SUB[best_prog_id])
        self.trainResult["candidates_history"].append(copy.deepcopy(envs.main_env.programs[best_prog_id]))
        if self.trainResult["candidates_history"][-1].include_free_const:
            self.best_free_const_values = envs.main_env.programs.free_const_values.values[best_prog_id]
            self.best_semi_free_const_values = envs.main_env.programs.semi_free_const_values.values[best_prog_id]
        else:
            self.best_free_const_values = envs.main_env.programs.free_const_values.init_val_tensor
            self.best_semi_free_const_values = envs.main_env.programs.semi_free_const_values.init_val_tensor
        self.trainResult["free_const_history"].append(
            self.best_free_const_values.detach().cpu().numpy().tolist())
        self.trainResult["semi_free_const_history"].append(
            self.best_semi_free_const_values.detach().cpu().numpy().tolist())
        self.trainResult["mean_R_history"].append(R.mean())
        self.trainResult["R_train_history"].append(R_train)
        self.trainResult["mean_R_train_history"].append(R_train.mean())
        self.trainResult["mean_R_train_sub_history"].append(R_train_sub.mean(axis=0))

    def update_epoch_different_expressions(self, R_train, keep, R, R_SUB, envs, update_number=1, need_simply=True):
        if not (R_train == 0.).all():  # not all zeros
            count = 0
            for i in range(len(keep)):
                if R[keep[i]] != 0.:
                    list_tokens_idx = list(envs.main_env.library.tokens_idx[keep[i]])
                    if list_tokens_idx not in self.trainResult[
                        "different_candidates_tokens_idx"]:  # np array have error
                        self.trainResult["different_candidates_history"].append(copy.deepcopy(envs.main_env.programs[keep[i]]))
                        try:
                            self.trainResult["different_candidates_infix_history"].append(str(
                                envs.main_env.programs[keep[i]].get_infix_sympy(
                                    do_simplify=need_simply if len(envs.main_env.programs[keep[i]].tokens) < simplify_True_length_limit else False,
                                    no_const_subtitution=True))) #这里不代入常数
                        except:
                            self.trainResult["different_candidates_infix_history"].append("error_in_simplify")
                        if len(envs.main_env.programs[keep[i]].tokens) >= simplify_True_length_limit:
                            print(f"The prefix expression is {len(envs.main_env.programs[keep[i]].tokens)}, too long don't simplify.")
                        self.trainResult["different_sub_rewards_history"].append(R_SUB[keep[i]])
                        self.trainResult["different_rewards_history"].append(R[keep[i]])
                        self.trainResult["different_candidates_tokens_idx"].append(list_tokens_idx)
                        if self.trainResult["different_candidates_history"][-1].include_free_const:
                            self.best_free_const_values = copy.deepcopy(envs.main_env.programs.free_const_values.values[keep[i]])
                            self.best_semi_free_const_values = copy.deepcopy(envs.main_env.programs.semi_free_const_values.values[keep[i]])
                        else:
                            self.best_free_const_values = copy.deepcopy(envs.main_env.programs.free_const_values.init_val_tensor)
                            self.best_semi_free_const_values = copy.deepcopy(envs.main_env.programs.semi_free_const_values.init_val_tensor)
                        self.trainResult["different_free_const_values_history"].append(
                            self.best_free_const_values.detach().cpu().numpy().tolist())
                        self.trainResult["different_semi_free_const_values_history"].append(
                            self.best_semi_free_const_values.detach().cpu().numpy().tolist())
                        # add more
                        count += 1
                        if count > update_number:
                            break
                        # break
                else:
                    break

    def update_evolution_best_program(self, R, R_SUB, envs, y, X_numpy, device):
        X_tensor = torch.tensor(X_numpy).to(device,
                                            dtype=torch.float32)
        if not self.trainResult["best_R_history"] or R.max() > self.trainResult["best_R_history"][
            -1]:
            self.trainResult["best_R_history"].append(R.max())
            self.trainResult["best_sub_R_history"].append(R_SUB[self.best_prog_id]) # 是根据best_R_history来更新sub的
            self.trainResult["best_program"] = copy.deepcopy(envs.main_env.programs[self.best_prog_id])
            best_a = envs.main_env.programs[self.best_prog_id](X_tensor)
            self.best_a_loss = MSE_loss(y_pred=best_a, y_target=y).item()

            # _, _, best_a, best_v, best_x = ALL_REWARD_FUNCTION_VELOCITY_TRAJECTORY(
            #     self.trainResult["best_program"], self.ngsim_args,
            #     envs.main_env.programs.free_const_values.values[self.best_prog_id],
            #     envs.main_env.programs.semi_free_const_values.values[self.best_prog_id])
            # self.best_a_loss = MSE_loss(y_pred=best_a, y_target=y).item()
            # self.best_v_loss = MSE_loss(y_pred=best_v, y_target=self.ngsim_args["v_obs"]).item()
            # self.best_x_loss = MSE_loss(y_pred=best_x, y_target=self.ngsim_args["x_obs"]).item()
            # if self.bool_args['bool_plot_intermediate_process'] and R.max() > self.bool_args[
            #     'plot_intermediate_process_limit']:
            #     plt.plot(best_a.detach().numpy(), label="a")
            #     plt.plot(self.ngsim_args["a_obs"][:-1].detach().numpy(), label="a_obs")
            #     plt.legend()
            #     plt.show()
            #
            #     plt.plot(best_v.detach().numpy(), label="v")
            #     plt.plot(self.ngsim_args["v_obs"].detach().numpy(), label="v_obs")
            #     plt.legend()
            #     plt.show()
            #
            #     plt.plot(best_x.detach().numpy(), label="x")
            #     plt.plot(self.ngsim_args["x_obs"].detach().numpy(), label="x_obs")
            #     plt.legend()
            #     plt.show()
        else:
            self.trainResult["best_R_history"].append(self.trainResult["best_R_history"][-1])
            self.trainResult["best_sub_R_history"].append(self.trainResult["best_sub_R_history"][-1])

    def update_evolution_similarity_best_sub_reward(self, R_SUB, ):
        if not self.trainResult["similarity_best_R_sub_history"] or R_SUB[:, 1].max() > self.trainResult["similarity_best_R_sub_history"][-1][1]:
            self.trainResult["similarity_best_R_sub_history"].append([0, R_SUB[:, 1].max(), 0]) # 是根据sub 1：similarity本身的来更新sub的，从而记录了最相似的
        else:
            self.trainResult["similarity_best_R_sub_history"].append(self.trainResult["similarity_best_R_sub_history"][-1])

    def evolutionBest_append_trainResult(self, best_prog_id):
        self.bestInEvolutions["best_R_history"].append(
            self.trainResult["different_rewards_history"][best_prog_id])
        self.bestInEvolutions["best_sub_R_history"].append(
            self.trainResult["different_sub_rewards_history"][best_prog_id].tolist())
        self.bestInEvolutions["best_program_history"].append(
            copy.deepcopy(self.trainResult["different_candidates_history"][best_prog_id]))
        self.bestInEvolutions["best_free_const_values_history"].append(
            self.trainResult["different_free_const_values_history"][best_prog_id])
        self.bestInEvolutions["best_semi_free_const_values_history"].append(
            self.trainResult["different_semi_free_const_values_history"][best_prog_id])
