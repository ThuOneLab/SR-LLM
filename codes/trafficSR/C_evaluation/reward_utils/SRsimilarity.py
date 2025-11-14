import math
import numpy as np
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import Token, COMMUTATIVE_FUNCTIONS
from codes.trafficSR.A_sampling.env_composition.SRlibrary import ILLEGAL_RELATIVE_INDEX

operator_category = {
    "add": ["sub"],
    "sub": ["add"],
    "mul": ["div"],
    "div": ["mul"],
    "n2": ["n3", "n4"],
    "n3": ["n2", "n4"],
    "n4": ["n2", "n3"],
    "sin": ["cos", "tan"],
    "cos": ["sin", "tan"],
    "tan": ["sin", "cos"],
}
special_operator = {"n2": 1.,
                    "n3": 4.,
                    "n4": 4.,
                    "sin": 2.,
                    "cos": 2.,
                    "tan": 2.,
                    "exp": 4.,
                    "log": 4., }

DEFAULT_SIMILARITY_SETTING = {
    "free_constant_different": 0.,
    "free_constant_unit_same": 0.2,
    "free_constant_same": 1.0,

    "fixed_constant_all_different": 0.,
    "fixed_constant_unit_same": 0.15,
    "fixed_constant_same": 1.0,

    "variable_different": 0,
    "variable_unit_same": 0.2,
    "variable_same": 1.,

    "operator_different": 0.,
    "operator_same_category": 0.1,
    "operator_same": 1.,
    # "special_operator_same": 2.,

    "free_fixed_const_same_unit": 0.15,

    "const_variable_same_unit": 0.1,

    "boost_weight": 1.25,  # function more important than others
    "depth_penalty": -0.5,  # -0.4:0.67; -0.5:0.607
}


class Similarity:
    def __init__(self, library, anchor_prog_id=-1, similarity_reward=None, similarity_new_version=True):
        self.library = library
        if similarity_reward is None:
            self.similarity_reward = DEFAULT_SIMILARITY_SETTING
        else:
            self.similarity_reward = similarity_reward
        self.similarity_new_version = similarity_new_version
        self.anchor_prog_init(anchor_prog_id)

    def anchor_prog_init(self, anchor_prog_id):
        if anchor_prog_id >= 0:
            self.super_parent_height = self.library.super_parent_info.height[anchor_prog_id, :]
            self.super_parent_children_index = self.library.super_parent_info.children_index[anchor_prog_id, :, :]
            self.super_parent_children_number = self.library.super_parent_info.children_number[anchor_prog_id, :]
            self.super_parent_children_end = self.library.super_parent_info.children_end[anchor_prog_id, :, :]
            self.super_parent_children_length = self.library.super_parent_info.children_length[anchor_prog_id, :, :]
            self.get_beta_weights_of_children()
            self.anchor_prog = self.library.super_parent_info.anchor_prog[anchor_prog_id].tokens

    def importance_length(self, length1, length2):
        log1 = math.log(length1 + 1)
        log2 = math.log(length2 + 1)
        sum_weight = 2 / (2 + self.similarity_reward["boost_weight"])
        weight1 = sum_weight * log1 / (log1 + log2)
        weight2 = sum_weight * log2 / (log1 + log2)
        return weight1, weight2

    def get_beta_weights_of_children(self):
        mask_super_parent_have_one_child = self.super_parent_children_number == 1
        mask_super_parent_have_two_child = self.super_parent_children_number == 2
        self.super_parent_beta_weight = np.zeros_like(self.super_parent_children_length, dtype=np.float32)
        self.super_parent_beta_weight[mask_super_parent_have_one_child, 0] = 1 / (
                1 + self.similarity_reward["boost_weight"])
        for i in np.where(mask_super_parent_have_two_child)[0]:
            self.super_parent_beta_weight[i, 0], self.super_parent_beta_weight[i, 1] = self.importance_length(
                self.super_parent_children_length[i, 0], self.super_parent_children_length[i, 1])

    def height1(self, token: Token, superparent_token: Token):  # for two token
        if token.token_type == "free_const" or token.token_type == "fixed_const":
            if superparent_token.token_type == "variable" and (token.phy_units == superparent_token.phy_units).all():
                return self.similarity_reward["const_variable_same_unit"]
        if superparent_token.token_type == "free_const" or superparent_token.token_type == "fixed_const":
            if token.token_type == "variable" and (token.phy_units == superparent_token.phy_units).all():
                return self.similarity_reward["const_variable_same_unit"]
        if (superparent_token.token_type == "free_const" and token.token_type == "fixed_const") or (
                superparent_token.token_type == "fixed_const" and token.token_type == "free_const"):
            if (token.phy_units == superparent_token.phy_units).all():
                return self.similarity_reward["free_fixed_const_same_unit"]

        if token.token_type == "free_const" and superparent_token.token_type == "free_const":
            if (token.phy_units == superparent_token.phy_units).all():
                if token.token_name == superparent_token.token_name:
                    return self.similarity_reward["free_constant_same"]
                else:
                    return self.similarity_reward["free_constant_unit_same"]
            else:
                return self.similarity_reward["free_constant_different"]
        if token.token_type == "fixed_const" and superparent_token.token_type == "fixed_const":
            if (token.phy_units == superparent_token.phy_units).all():
                if token.token_name == superparent_token.token_name:
                    return self.similarity_reward["fixed_constant_same"]
                else:
                    return self.similarity_reward["fixed_constant_unit_same"]
            else:
                return self.similarity_reward["fixed_constant_all_different"]
        if token.token_type == "variable" and superparent_token.token_type == "variable":
            if (token.phy_units == superparent_token.phy_units).all():
                if token.token_name == superparent_token.token_name:
                    return self.similarity_reward["variable_same"]
                else:
                    return self.similarity_reward["variable_unit_same"]
            else:
                return self.similarity_reward["variable_different"]
        if token.token_type == "operator" and superparent_token.token_type == "operator":
            if token.token_name == superparent_token.token_name:
                if token.token_name in special_operator:
                    return special_operator[token.token_name]
                else:
                    return self.similarity_reward["operator_same"]
            elif token.token_name in operator_category and superparent_token.token_name in operator_category[
                token.token_name]:
                return self.similarity_reward["operator_same_category"]
            else:
                return self.similarity_reward["operator_different"]
        return 0.

    def compute_sum_reward_old_version(self, prog_index, superparent_index):
        p = self.chidren_number[prog_index]
        q = self.super_parent_children_number[superparent_index]
        minpq = min(p, q)
        alpha = self.similarity_reward["boost_weight"] / (
                p + self.similarity_reward["boost_weight"])
        beta = 1. / (p + self.similarity_reward["boost_weight"])
        sum_reward = alpha * self.height1(self.prog[prog_index],
                                          self.anchor_prog[superparent_index])  # it is function
        if self.prog[prog_index].token_name in COMMUTATIVE_FUNCTIONS or self.anchor_prog[
            superparent_index].token_name in COMMUTATIVE_FUNCTIONS:
            reward = np.zeros((2, 2))
            for i in range(p):
                for j in range(q):
                    reward[i][j] = self.compute_tree_reward(self.children_index[prog_index][i],
                                                            self.super_parent_children_index[superparent_index][j])
            if p == q and p == 2:
                sum_reward = sum_reward + beta * max(reward[0][0] + reward[1][1], reward[0][1] + reward[1][0])
            else:
                sum_reward = sum_reward + beta * np.max(reward)
        else:
            for i in range(minpq):
                sum_reward = sum_reward + beta * self.compute_tree_reward(self.children_index[prog_index][i],
                                                                          self.super_parent_children_index[
                                                                              superparent_index][i])
        return p, q, sum_reward

    def compute_sum_reward(self, prog_index, superparent_index):
        p = self.chidren_number[prog_index]
        q = self.super_parent_children_number[superparent_index]
        minpq = min(p, q)
        # should be anchor q, because need to consider how many children in anchor
        alpha = self.similarity_reward["boost_weight"] / (
                q + self.similarity_reward["boost_weight"])
        # beta = 1. / (q + self.similarity_reward["boost_weight"])
        beta_weights = self.super_parent_beta_weight[superparent_index]
        sum_reward = alpha * self.height1(self.prog[prog_index],
                                          self.anchor_prog[superparent_index])  # it is function
        if self.prog[prog_index].token_name in COMMUTATIVE_FUNCTIONS or self.anchor_prog[
            superparent_index].token_name in COMMUTATIVE_FUNCTIONS:
            reward = np.zeros((2, 2))
            for i in range(p):
                for j in range(q):
                    reward[i][j] = self.compute_tree_reward(self.children_index[prog_index][i],
                                                            self.super_parent_children_index[superparent_index][j])
            if p == 2 and q == 2:  # the weight is for q
                sum_reward = sum_reward + max(beta_weights[0] * reward[0][0] + beta_weights[1] * reward[1][1],
                                              beta_weights[1] * reward[0][1] + beta_weights[0] * reward[1][0])
            else:  # for commutative function, the order of children is not important
                # for example, 5+2 and n2(5), just one reward will be added
                sum_reward = sum_reward + max(beta_weights[0] * reward[0][0], beta_weights[1] * reward[1][1],
                                              beta_weights[1] * reward[0][1], beta_weights[0] * reward[1][0])
        else:
            for i in range(minpq):
                sum_reward = sum_reward + beta_weights[i] * self.compute_tree_reward(self.children_index[prog_index][i],
                                                                                     self.super_parent_children_index[
                                                                                         superparent_index][i])
        return p, q, sum_reward

    # for two tokens_list that have same height
    def same_height(self, prog_index: int, superparent_index: int):
        if self.height[prog_index] == 1 and self.super_parent_height[superparent_index] == 1:
            return self.height1(self.prog[prog_index], self.anchor_prog[superparent_index])
        # main-tree compute
        if self.similarity_new_version:
            p, q, sum_reward = self.compute_sum_reward(prog_index, superparent_index)
        else:
            p, q, sum_reward = self.compute_sum_reward_old_version(prog_index, superparent_index)
        return sum_reward

    def different_height(self, prog_index: int, superparent_index: int):
        # main-tree compute
        if self.similarity_new_version:
            p, q, sum_reward = self.compute_sum_reward(prog_index, superparent_index)
        else:
            p, q, sum_reward = self.compute_sum_reward_old_version(prog_index, superparent_index)
        # subtree compute: depth_penalty
        if self.height[prog_index] > self.super_parent_height[prog_index]:  # use T1 subtree to compare with T2
            for i in range(p):
                sum_reward = max(sum_reward,
                                 math.exp(self.similarity_reward["depth_penalty"]) * self.compute_tree_reward(
                                     self.children_index[prog_index][i], superparent_index))
        else:  # use T2 subtree to compare with T1
            for i in range(q):
                sum_reward = max(sum_reward,
                                 math.exp(self.similarity_reward["depth_penalty"]) * self.compute_tree_reward(
                                     prog_index,
                                     self.super_parent_children_index[
                                         superparent_index][
                                         i]))
        return sum_reward

    def compute_tree_reward(self, prog_tree_start, superparent_tree_start):
        if prog_tree_start >= ILLEGAL_RELATIVE_INDEX or superparent_tree_start >= ILLEGAL_RELATIVE_INDEX:  # illegal index, such as sqrt-mul in different height
            self.reward_matrix[prog_tree_start, superparent_tree_start] = 0.
            return 0.

        if prog_tree_start >= len(self.prog) or superparent_tree_start >= len(
                self.anchor_prog):  # illegal index, such as sqrt-mul in different height
            return 0.

        if self.reward_matrix[prog_tree_start, superparent_tree_start] >= 0:
            return self.reward_matrix[prog_tree_start, superparent_tree_start]

        if self.height[prog_tree_start] == 1 and self.super_parent_height[superparent_tree_start] == 1:
            self.reward_matrix[prog_tree_start, superparent_tree_start] = self.height1(
                self.prog[prog_tree_start], superparent_token=self.anchor_prog[superparent_tree_start])
        elif self.height[prog_tree_start] == self.super_parent_height[superparent_tree_start]:
            self.reward_matrix[prog_tree_start, superparent_tree_start] = self.same_height(prog_tree_start,
                                                                                           superparent_tree_start)
        else:
            self.reward_matrix[prog_tree_start, superparent_tree_start] = self.different_height(prog_tree_start,
                                                                                                superparent_tree_start)
        return self.reward_matrix[prog_tree_start, superparent_tree_start]

    def prog_init(self, program, prog_id: int, first_input=True):
        self.prog = program
        # have combination and first input into similarity
        if self.library.combination_info.have_combination[prog_id] and first_input:
            self.library.update_combination_height_children(prog_id)
            self.library.get_prog_children_end_and_length(prog_id)
        self.height = self.library.height[prog_id]
        self.children_index = self.library.children_index[prog_id]
        self.chidren_number = self.library.children_number[prog_id]
        self.children_end = self.library.children_end[prog_id]

    def compute(self, program, prog_id: int, first_input=True):
        # Compute the similarity between the program and the superparent
        # program: SRProgram
        # return: similarity score
        self.prog_init(program.tokens, prog_id, first_input=first_input)
        self.reward_matrix = np.full(shape=(len(self.prog), len(self.anchor_prog)), fill_value=-1.0,
                                     dtype=np.float32)

        if self.height[0] == 1 and self.super_parent_height[0] == 1:
            self.reward_matrix[0, 0] = self.height1(token=self.prog[0],
                                                    superparent_token=self.anchor_prog[0])
        elif self.height[0] == self.super_parent_height[0]:
            self.reward_matrix[0, 0] = self.same_height(0, 0)
        else:
            self.reward_matrix[0, 0] = self.different_height(0, 0)
        return self.reward_matrix[0, 0]


class Similarity_numerical_score(Similarity):
    def __init__(self, combinations_prog, library, similarity_reward=None, score=None, combinations_num=0):
        super().__init__(library, anchor_prog_id=-1, similarity_reward=similarity_reward)
        self.combinations_prog = combinations_prog
        self.library = library
        if similarity_reward is None:
            self.similarity_reward = DEFAULT_SIMILARITY_SETTING
        else:
            self.similarity_reward = similarity_reward
        self.score = score  # score result
        self.anchor_num = len(self.library.super_parent_info.anchor_prog)
        self.combinations_num = combinations_num
        self.anchor_length = [len(self.library.super_parent_info.anchor_prog[anchor_prog_id]) for anchor_prog_id
                              in range(self.anchor_num)]
        self.combinations_length = [len(self.combinations_prog[key]["prog"]) for key in
                                    self.combinations_prog]

    def obtain_possible_show_index_in_expressions(self):
        # Obtain possible exact match indices in anchor prog
        possible_index = []
        prog_length = len(self.prog)
        if len(self.anchor_prog) < prog_length:
            return possible_index
        for i in range(0, len(self.anchor_prog) - prog_length + 1):
            if self.prog[0].token_name == self.anchor_prog[i].token_name and self.height[0] == \
                    self.super_parent_height[i] and self.super_parent_height[i + prog_length - 1] == 1:
                possible_index.append(i)
        return possible_index

    def bool_match_each_other(self, anchor_possible_index):
        self.reward_matrix = np.full(shape=(self.library.max_time_step, self.library.max_time_step), fill_value=-1.0,
                                     dtype=np.float32)
        self.reward_matrix[0, 0] = self.same_height(prog_index=0, superparent_index=anchor_possible_index)
        if self.reward_matrix[0, 0] == 1:
            return True
        else:
            return False

    def compute(self, prog_id: int, anchor_prog_id: int, first_input=True):
        # Compute the similarity between the program and the superparent
        # program: SRProgram
        # return: similarity score

        # init anchor_prog and prog
        self.anchor_prog_init(anchor_prog_id)
        items = list(self.combinations_prog.items())
        self.prog_init(self.combinations_prog[items[prog_id][0]]["prog"], prog_id, first_input=first_input)

        # Obtain possible exact match indices in anchor prog
        possible_index = self.obtain_possible_show_index_in_expressions()
        show_flag = False
        show_times = 0

        # Determine if they match each other
        for anchor_possible_index in possible_index:
            if self.bool_match_each_other(anchor_possible_index):
                show_flag = True
                show_times += 1
        return show_flag, show_times

    def update_numerical_score(self, reward_of_expressions):
        score_items = list(self.score.items())
        for combination_id in range(self.combinations_num):
            combination_name = score_items[combination_id][
                0]  # the i-th combination, but have anchor_length and anchor_reward
            self.score[combination_name]['combination_prefix'] = [token.representation for token in
                                                                  self.combinations_prog[combination_name]["prog"]]
            self.score[combination_name]['combination_length'] = self.combinations_length[combination_id]
            for anchor_id in range(self.anchor_num):
                if combination_id == 0:  # only update once, shorten print
                    self.score[combination_name]['anchor_length'].append(self.anchor_length[anchor_id])
                    self.score[combination_name]['anchor_reward'].append(reward_of_expressions[anchor_id])
                show_flag, show_times = self.compute(prog_id=combination_id,
                                                     anchor_prog_id=anchor_id,
                                                     first_input=True if anchor_id == 0 else False)
                if show_flag:
                    self.score[combination_name]['show_id_in_anchor'].append(anchor_id)
                    # else:
                    #     self.score[combination_name]['show_id_in_anchor'].append(-1)
                    self.score[combination_name]['show_times_in_anchor'].append(show_times)
                    self.score[combination_name]['combination_contribution'].append(
                        self.combinations_length[combination_id] * show_times / self.anchor_length[
                            anchor_id])  # show proportion
                    self.score[combination_name]['numerical_score'].append(
                        reward_of_expressions[anchor_id] *
                        self.score[combination_name]['combination_contribution'][-1])
