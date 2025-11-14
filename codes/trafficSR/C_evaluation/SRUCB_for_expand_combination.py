import numpy as np
class UCB_for_expand_combination:
    def __init__(self, k_best_of_symbols):
        self.UCB_for_expand_combination = {}
        self.k_best_of_symbols = k_best_of_symbols
        self.UCB_weight = 0.1
        self.UCB_weight_token = 0.15  # from 0.3(16:30)
        self.UCB_weight_combination = 0.1

    def init_UCB_info(self, env):
        self.consider_group = env.library.all_token_info_table.variable_group + env.library.all_token_info_table.combination_group
        self.consider_group = self.consider_group + [
            env.library.all_tokens_id_dict["alpha"]] if "alpha" in env.library.all_tokens_id_dict else self.consider_group
        for combination_id in self.consider_group:
            symbol = env.library.all_token_info_table.token_name_table[combination_id]
            if symbol not in self.UCB_for_expand_combination.keys():
                self.UCB_for_expand_combination[symbol] = {"n_selected_for_expand_times": 0,
                                                           "n_epochs_has_experienced": 0,
                                                           "UCB_score": 0,
                                                           "combination": True if combination_id in env.library.all_token_info_table.combination_group else False}

    def update_UCB_n_selected_times(self, symbol):
        self.UCB_for_expand_combination[symbol]["n_selected_for_expand_times"] += 1

    def update_UCB_n_epochs_has_experienced(self, symbol):
        self.UCB_for_expand_combination[symbol]["n_epochs_has_experienced"] += 1

    def update_UCB_score(self, symbol):
        symbol_UCB_item = self.UCB_for_expand_combination[symbol]
        # symbol_UCB_item["UCB_score"] = np.sqrt(self.UCB_weight *
        #                                        np.log(self.k_best_of_symbols * (
        #                                                symbol_UCB_item["n_epochs_has_experienced"] + 1)) / (
        #                                                symbol_UCB_item["n_selected_for_expand_times"] + 1))

        # different weight: weight for token larger
        # weight = self.UCB_weight_combination if symbol_UCB_item["combination"] else self.UCB_weight_token
        # if (not symbol_UCB_item["combination"]) and symbol_UCB_item["n_selected_for_expand_times"] == 0:
        #     symbol_UCB_item["UCB_score"] = 1. # for variable, need initialize at least once
        # else:
        #     symbol_UCB_item["UCB_score"] = np.sqrt(weight *
        #                                        np.log(self.k_best_of_symbols * (
        #                                                symbol_UCB_item["n_epochs_has_experienced"] + 1)) / (
        #                                                symbol_UCB_item["n_selected_for_expand_times"] + 1))

        weight = self.UCB_weight_combination if symbol_UCB_item["combination"] else self.UCB_weight_token
        symbol_UCB_item["UCB_score"] = np.sqrt(weight *
                                               np.log(self.k_best_of_symbols * (
                                                       symbol_UCB_item["n_epochs_has_experienced"] + 1)) / (
                                                       symbol_UCB_item["n_selected_for_expand_times"] + 1))

    def get_UCB_score(self, symbol):
        return self.UCB_for_expand_combination[symbol]["UCB_score"]

    def update_UCB_info(self, best_symbols, worst_combinations, env):
        # update all and best
        for combination_id in self.consider_group:
            combination_name = env.library.all_token_info_table.token_name_table[combination_id]
            if combination_name in best_symbols.keys():
                self.update_UCB_n_selected_times(combination_name)
            self.update_UCB_n_epochs_has_experienced(combination_name)
            self.update_UCB_score(combination_name)

        # delete worst
        filtered_dict = {k: v for k, v in self.UCB_for_expand_combination.items() if k not in worst_combinations.keys()}
        self.UCB_for_expand_combination = filtered_dict