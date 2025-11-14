import copy
import numpy as np
from statistics import mean
import codes.trafficSR.A_sampling.env_generator.SRenv as SRenv
from codes.trafficSR.A_sampling.env_composition.SRprograms import update_library
from codes.trafficSR.C_evaluation.reward_utils.SRsimilarity import Similarity_numerical_score
from codes.trafficSR.D_updation_by_LLM.Agent import Agent

class SRscore:
    def __init__(self, env: SRenv, different_expression, agent: Agent, reward_of_expressions: list):
        self.env = env
        self.score = {}
        self.consider_group = env.library.all_token_info_table.variable_group + env.library.all_token_info_table.combination_group
        self.consider_group = self.consider_group + [
            env.library.all_tokens_id_dict["alpha"]] if "alpha" in env.library.all_tokens_id_dict else self.consider_group
        self.init_score()
        self.different_expression = different_expression
        self.agent = agent
        self.combinations_num = 0
        self.combinations_info, self.combinations_prog = self.get_combinations()
        # update env library by input combinations
        if len(self.combinations_info) > 0:
            self.update_env_library()
        self.similarity_numerical_score = Similarity_numerical_score(combinations_prog=self.combinations_prog,
                                                                     library=self.env.library, similarity_reward=
                                                                     self.env.env_args["similarity_args"][
                                                                         "similarity_reward"], score=self.score,
                                                                     combinations_num=len(self.combinations_info))
        self.reward_of_expressions = reward_of_expressions

    def init_score(self):
        self.combination_names_group = []
        for combination_id in self.consider_group:
            combination_name = self.env.library.all_token_info_table.token_name_table[combination_id]
            if combination_id in self.env.library.all_token_info_table.combination_group:
                self.combination_names_group.append(combination_name)
            self.score[combination_name] = {}
            self.score[combination_name]['anchor_length'] = []
            self.score[combination_name]['anchor_reward'] = []
            self.score[combination_name]['show_id_in_anchor'] = []
            self.score[combination_name]['show_times_in_anchor'] = []
            self.score[combination_name]['combination_contribution'] = []
            self.score[combination_name]['numerical_score'] = []
            self.score[combination_name]['mean_numerical_score'] = 0
            self.score[combination_name]['UCB_score'] = 0
            self.score[combination_name]['semantic_score'] = 0
            self.score[combination_name]['integrated_score'] = 0

    def get_combinations(self):
        combinations_info, combinations_prog = {}, {}
        for combination_id in self.consider_group:
            combination_name = self.env.library.all_token_info_table.token_name_table[combination_id]
            combinations_info[combination_name] = {}
            combinations_info[combination_name]["infix_expression"] = self.env.all_tokens[
                combination_id].infix_expression
            combinations_info[combination_name]["description"] = self.env.all_tokens[
                combination_id].description
            combinations_info[combination_name]["units"] = self.env.all_tokens[
                combination_id].phy_units.tolist()
            combinations_info[combination_name]["tokens_idx"] = self.env.all_tokens[
                combination_id].tokens_idx  # list of tokens idx
            # list of token name

            combinations_prog[combination_name] = {}
            if combination_id in self.env.library.all_token_info_table.combination_group:
                combinations_info[combination_name]["type"] = "combination"
                self.combinations_num += 1
                combinations_info[combination_name]["tokens_name"] = [token.token_name for token in self.env.all_tokens[combination_id].tokens_list]
                combinations_prog[combination_name]["prog"] = self.env.all_tokens[
                    combination_id].tokens_list
            else:
                combinations_info[combination_name]["type"] = "variable"
                combinations_info[combination_name]["tokens_name"] = [self.env.all_tokens[combination_id].token_name]

                combinations_prog[combination_name]["prog"] = [
                    self.env.all_tokens[combination_id]]  # list of tokens

        return combinations_info, combinations_prog

    # for the combinations, input them and get the similarity need
    def update_env_library(self):
        combination_max_length = 0
        for key in self.combinations_info.keys():
            if len(self.combinations_info[key]["tokens_idx"]) > combination_max_length:
                combination_max_length = len(self.combinations_info[key]["tokens_idx"])
        # next(iter(self.combinations)) # get key of the first combination
        first_combination_value = self.combinations_info.get(next(iter(self.combinations_info)))

        for time_step in range(combination_max_length):
            new_tokens_idx = np.full((self.env.batch_size),
                                     fill_value=first_combination_value["tokens_idx"][-1],
                                     # last token is not op (self.all_tokens_id_dict["end"] not in dict)
                                     dtype=int)
            # Fill in the token_ist of the combination and input it into env to update the library.
            # Thus constructing an nev where all valid programs are a combination.
            for combination_j, (k, v) in enumerate(self.combinations_info.items()):  # k is combination_name;
                new_tokens_idx[combination_j] = v["tokens_idx"][time_step] if time_step < len(
                    v["tokens_idx"]) else self.env.library.all_tokens_id_dict["end"]
            update_library(self.env.library, new_tokens_idx=new_tokens_idx)
        self.env.library.get_height()
        self.env.library.get_children_end_and_length()

    def get_numerical_score(self):
        self.similarity_numerical_score.update_numerical_score(reward_of_expressions=self.reward_of_expressions)

    def get_UCB_score(self, combination_UCB):
        for symbol_name in self.score.keys():
            self.score[symbol_name]['UCB_score'] = combination_UCB.get_UCB_score(symbol_name)

    def get_semantic_score(self, evolution=0, have_get_semantic_score=False):
        self.agent.get_semantic_score_md(self.combinations_info, evolution=evolution,
                                         have_get_semantic_score=have_get_semantic_score)
        # get semantic_score_dict
        code_blocks = self.agent.extract_python_code_blocks(
            f"{self.agent.trainResultFolder}semantic_score_dict{evolution}.md")
        semantic_score_dict = eval(code_blocks[-1])
        # update semantic score
        for combination_name in semantic_score_dict.keys():
            self.score[combination_name]['semantic_score'] = semantic_score_dict[combination_name]["semantic_score"]
        return True

    def get_integrated_score(self):
        for combination_name in self.score.keys():
            # 以前还会加上semantic_score
            # self.score[combination_name]['integrated_score'] = mean(self.score[combination_name]['numerical_score']) + \
            #                                                    self.score[combination_name]['semantic_score']
            numerical_score = [score for score in self.score[combination_name]['numerical_score'] if score > 0]
            if len(numerical_score) > 0:
                self.score[combination_name]['mean_numerical_score'] = mean(numerical_score)

            if combination_name in self.combination_names_group:
                # if len(numerical_score) > 0:  # only elite combination can get UCB score
                self.score[combination_name]['integrated_score'] = self.score[combination_name][
                                                                       'mean_numerical_score'] + \
                                                                   self.score[combination_name]['UCB_score']
            else:  # for variable explore, all get UCB score
                self.score[combination_name]['integrated_score'] = self.score[combination_name][
                                                                       'mean_numerical_score'] + \
                                                                   self.score[combination_name]['UCB_score']

    # useless
    def select_best_combinations(self, top_k=5):
        sorted_score = sorted(self.score.items(), key=lambda x: x[1]['integrated_score'],
                              reverse=True)  # sorted_score[i] means the i-th best combination
        best_combinations = {}
        best_combinations["new_combinations"] = {}
        for i in range(top_k):
            best_combinations["new_combinations"][sorted_score[i][0]] = {}
            
            dict_item = best_combinations["new_combinations"][sorted_score[i][0]]
            dict_item["infix_expression"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['infix_expression'])
            dict_item["description"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['description'])
            dict_item["units"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['units'])
            dict_item["prefix_expression"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['tokens_name'])
        return best_combinations

    #获得最佳和最差组合，用于后续
    def select_best_worst_combinations_rag(self, top_k=2, worst_k=1, k_best_of_expressions=5, directly_built=None):
        if directly_built is None:
            directly_built = []
        sorted_score = sorted(self.score.items(), key=lambda x: x[1]['integrated_score'],
                              reverse=True)  # sorted_score[i] means the i-th best combination
        '''get best combinations'''
        best_symbols, best_expressions = {}, []
        for i in range(min(top_k, len(sorted_score))):
            best_symbols[sorted_score[i][0]] = {}
            dict_item = best_symbols[sorted_score[i][0]]
            dict_item["name"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['infix_expression'])
            dict_item["description"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['description'])
            dict_item["units"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['units'])
            dict_item["prefix_expression"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['tokens_name'])
            dict_item["type"] = copy.deepcopy(
                self.combinations_info[sorted_score[i][0]]['type'])

            # mask_show_in_anchor = np.array(self.score[sorted_score[i][0]]['show_id_in_anchor']) >= 0
            expressions_best_symbol_appeared = self.extract_expression_info(
                self.score[sorted_score[i][0]]['show_id_in_anchor'],
                k_best_of_expressions=k_best_of_expressions)
            best_expressions.append(expressions_best_symbol_appeared)  # list of dict

        sorted_score = sorted(self.score.items(), key=lambda x: x[1]['mean_numerical_score'],
                              reverse=True)  # sorted_score[i] means the i-th best combination
        '''get worst combinations'''
        worst_combinations = {}
        number_worst_have_selected = 0
        if worst_k + top_k < int(self.combinations_num / 3): #说明combination过多
            for i in range(len(sorted_score) - 1, 0, -1):
                # only the not-in-best combination, and not the first time
                if self.combinations_info[sorted_score[i][0]]['type'] == "combination" and sorted_score[i][
                    0] not in best_symbols.keys() and self.score[sorted_score[i][0]]['UCB_score'] > 0 and \
                        self.combinations_info[sorted_score[i][0]][
                            'infix_expression'] not in directly_built:
                    worst_combinations[sorted_score[i][0]] = {}
                    dict_item = worst_combinations[sorted_score[i][0]]
                    dict_item["name"] = copy.deepcopy(
                        self.combinations_info[sorted_score[i][0]]['infix_expression'])
                    dict_item["description"] = copy.deepcopy(
                        self.combinations_info[sorted_score[i][0]]['description'])
                    dict_item["units"] = copy.deepcopy(
                        self.combinations_info[sorted_score[i][0]]['units'])
                    dict_item["prefix_expression"] = copy.deepcopy(
                        self.combinations_info[sorted_score[i][0]]['tokens_name'])
                    dict_item["type"] = copy.deepcopy(
                        self.combinations_info[sorted_score[i][0]]['type'])
                    number_worst_have_selected += 1
                if number_worst_have_selected >= worst_k:
                    break
        # self.print_score_result()
        return best_symbols, worst_combinations, best_expressions

    def print_score_result(self):
        print("score result:")
        for i, (key, value) in enumerate(self.score.items()):
            print(f"{key}: {value}")

    def extract_expression_info(self, mask_show_in_anchor, k_best_of_expressions=5):
        new_dict = {}
        count = 0
        for i, (key, value) in enumerate(self.different_expression.items()):
            if i in mask_show_in_anchor:  # only extract show anchor
                new_dict[key] = {
                    'prefix_expression': value.get('prefix_expression', None),
                    'infix_expression': value.get('infix_expression', None),
                    'reward': value.get('sum_reward', None)
                }
                count += 1
            if count >= k_best_of_expressions:
                break
        return new_dict

    def __str__(self):
        return 'SRscore{score=' + str(self.score) + '}'
