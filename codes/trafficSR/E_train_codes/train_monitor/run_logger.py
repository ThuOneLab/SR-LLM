import numpy as np
import pandas as pd
import os


class RunLogger:
    """
    Custom logger function.
    """

    def __init__(self, save_folder=None, do_save=False, pareto_type="acceleration"):
        self.save_folder = save_folder
        self.do_save = do_save
        assert pareto_type in ["acceleration", "velocity", "displacement"], "Unknown pareto type"
        self.pareto_type = pareto_type
        self.initialize()

    def initialize(self):

        # Epoch specific
        self.epoch = None

        self.overall_max_R_history = []
        self.overall_best_prog_str_history = []
        self.hall_of_fame = []

        self.epochs_history = []
        self.loss_history = []

        self.mean_R_train_history = []
        self.mean_R_history = []
        self.max_R_history = []

        self.R_history = []
        self.R_history_train = []

        self.best_prog_epoch_str_history = []
        self.best_prog_complexity_history = []

        self.best_prog_epoch_str_prefix_history = []
        self.overall_best_prog_str_prefix_history = []

        self.best_prog_epoch_free_const_history = []
        self.overall_best_prog_free_const_history = []

        self.mean_complexity_history = []

        self.n_physical = []
        self.n_rewarded = []
        self.lengths_of_physical = []
        self.lengths_of_unphysical = []

    def log(self, epoch, batch, model, rewards, rewards_sub, keep, notkept, loss_val):

        # Epoch specific
        self.epoch = epoch
        # self.R       = rewards
        self.R_SUB = rewards_sub
        if self.pareto_type == "acceleration":
            self.R = rewards_sub[:, 2]
        elif self.pareto_type == "velocity":
            self.R = rewards_sub[:, 4]
        elif self.pareto_type == "displacement":
            self.R = rewards_sub[:, 3]
        else:
            raise ValueError("Unknown pareto type")
        self.batch = batch
        self.keep = keep
        self.notkept = notkept
        best_prog_idx_epoch = rewards.argmax()
        self.best_prog_epoch = batch.programs[best_prog_idx_epoch]
        self.programs_epoch = batch.library.tokens_idx

        if epoch == 0:
            self.free_const_names = [tok.__str__() for tok in self.batch.library.all_token_info_table.token_name_table[
                self.batch.library.all_token_info_table.free_const_group]]
            self.overall_max_R_history = [rewards.max()]
            self.hall_of_fame = [batch.programs[best_prog_idx_epoch]]
        if epoch > 0:
            if rewards.max() > np.max(self.overall_max_R_history):
                self.overall_max_R_history.append(rewards.max())
                self.hall_of_fame.append(batch.programs[best_prog_idx_epoch])
            else:
                self.overall_max_R_history.append(self.overall_max_R_history[-1])

        self.epochs_history.append(epoch)
        self.loss_history.append(loss_val)

        self.mean_R_train_history.append(rewards[keep].mean())
        self.mean_R_history.append(rewards.mean())
        self.max_R_history.append(rewards.max())

        self.R_history.append(rewards)
        self.R_history_train.append(rewards[keep])

        self.best_prog_epoch_str_history.append(self.best_prog_epoch.get_infix_notation())
        self.overall_best_prog_str_history.append(self.best_prog.get_infix_notation())

        self.best_prog_epoch_str_prefix_history.append(self.best_prog_epoch.__str__())
        self.overall_best_prog_str_prefix_history.append(self.best_prog.__str__())

        # Logging free const as str of a list
        self.best_prog_epoch_free_const_history.append(
            self.best_prog_epoch.free_const_values.detach().cpu().numpy().__str__())
        self.overall_best_prog_free_const_history.append(
            self.best_prog.free_const_values.detach().cpu().numpy().__str__())

        self.best_prog_complexity_history.append(batch.programs.valid_lengths[best_prog_idx_epoch])
        self.mean_complexity_history.append(batch.programs.valid_lengths.mean())

        self.R_history_array = np.array(self.R_history)
        self.R_history_train_array = np.array(self.R_history_train)

        self.n_physical.append(self.batch.library.is_physical.sum())
        self.n_rewarded.append((rewards > 0.).sum())
        self.lengths_of_physical.append(self.batch.programs.valid_lengths[self.batch.library.is_physical])
        self.lengths_of_unphysical.append(self.batch.programs.valid_lengths[~self.batch.library.is_physical])

        self.pareto_logger()

        # Saving log
        if self.do_save:
            self.save_log()

    def save_log(self):

        columns = ['epoch', 'reward', 'complexity', 'length', 'is_physical', 'is_elite', 'program', "program_prefix"]
        # Columns for free const names
        columns += self.free_const_names

        # Initial df
        if self.epoch == 0:
            df0 = pd.DataFrame(columns=columns)
            df0.to_csv(os.path.join(self.save_folder, "log.log"), index=False)

        # Current batch log
        is_elite = np.full(self.batch.batch_size, False)
        is_elite[self.keep] = True
        programs_str = []
        for i in range(self.batch.programs.batch_size):
            if self.batch.library.have_completed[i] and self.batch.library.is_physical[i]:
                programs_str.append(self.batch.programs[i].get_infix_notation())
            else:
                programs_str.append("")
        programs_str=np.array(programs_str)

        df = pd.DataFrame()
        df["epoch"] = np.full(self.batch.batch_size, self.epoch)
        df["reward"] = self.R
        df["complexity"] = self.batch.programs.valid_lengths
        df["length"] = self.batch.programs.valid_lengths
        df["is_physical"] = self.batch.library.is_physical
        df["is_elite"] = is_elite
        df["program"] = programs_str
        # df["program_prefix"] = self.batch.programs.get_programs_array()

        # Exporting free constants
        # free_const = self.batch.programs.free_consts.values.detach().cpu().numpy()
        # for i in range(len(self.free_const_names)):
        #     name   = self.free_const_names[i]
        #     const  = free_const[:, i]
        #     df[name] = const

        # Saving current df
        df.to_csv(os.path.join(self.save_folder, "log.log"), mode='a', index=False, header=False)

        return None

    def pareto_logger(self, ):
        curr_complexities = self.batch.programs.valid_lengths
        curr_rewards = self.R
        curr_batch = self.batch

        # Init
        if self.epoch == 0:
            self.pareto_complexities = np.arange(0, 10 * curr_batch.max_time_step)
            self.pareto_rewards = np.full(shape=(self.pareto_complexities.shape), fill_value=np.nan)
            self.pareto_programs = np.full(shape=(self.pareto_complexities.shape), fill_value=None, dtype=object)

        # Update with current epoch info
        for i, c in enumerate(self.pareto_complexities):
            # Idx in batch of programs having complexity c
            arg_have_c = np.argwhere(curr_complexities.round() == c)
            if len(arg_have_c) > 0:
                # Idx in batch of the program having complexity c and having max reward
                arg_have_c_and_max = arg_have_c[curr_rewards[arg_have_c].argmax()]
                # Max reward of this program
                max_r_at_c = curr_rewards[arg_have_c_and_max]
                # If reward > currently max reward for this complexity or empty, replace
                if self.pareto_rewards[i] <= max_r_at_c or np.isnan(self.pareto_rewards[i]):
                    self.pareto_programs[i] = curr_batch.programs[arg_have_c_and_max[0]]
                    self.pareto_rewards[i] = max_r_at_c

    def get_pareto_front(self, ):
        # Postprocessing
        # Keeping only valid pareto candidates
        mask_pareto_valid = (~np.isnan(self.pareto_rewards)) & (self.pareto_rewards > 0)
        pareto_rewards_valid = self.pareto_rewards[mask_pareto_valid]
        pareto_programs_valid = self.pareto_programs[mask_pareto_valid]
        pareto_complexities_valid = self.pareto_complexities[mask_pareto_valid]
        # Computing front
        if len(pareto_rewards_valid) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        pareto_front_r = [pareto_rewards_valid[0]]
        pareto_front_programs = [pareto_programs_valid[0]]
        pareto_front_complexities = [pareto_complexities_valid[0]]
        for i, r in enumerate(pareto_rewards_valid):
            # Only keeping candidates with higher reward than candidates having a smaller complexity
            if r > pareto_front_r[-1]:
                pareto_front_r.append(r)
                pareto_front_programs.append(pareto_programs_valid[i])
                pareto_front_complexities.append(pareto_complexities_valid[i])

        pareto_front_complexities = np.array(pareto_front_complexities)
        pareto_front_programs = np.array(pareto_front_programs)
        pareto_front_r = np.array(pareto_front_r)
        ngsim_args = self.batch.ngsim_args
        if self.pareto_type == "acceleration":
            pareto_front_rmse = ((1 / pareto_front_r) - 1) * ngsim_args["a_obs"].numpy().std()
        elif self.pareto_type == "velocity":
            pareto_front_rmse = ((1 / pareto_front_r) - 1) * ngsim_args["v_obs"].numpy().std()
        elif self.pareto_type == "displacement":
            pareto_front_rmse = ((1 / pareto_front_r) - 1) * ngsim_args["x_obs"].numpy().std()
        pareto_front_rmse = ((1 / pareto_front_r) - 1) * pareto_front_rmse.std()

        return pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse

    @property
    def best_prog(self):
        return self.hall_of_fame[-1]
