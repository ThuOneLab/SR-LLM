import copy
import numpy as np
from codes.trafficSR.A_sampling.env_composition.SRprogram import SRprogram
from codes.trafficSR.A_sampling.env_composition.SRprograms import update_library
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import UNITS_VECTOR_SIZE, TYPE_KIND_NUMBER
from codes.trafficSR.A_sampling.env_composition.SRlibrary import SRlibrary, ILLEGAL_RELATIVE_INDEX


class Token_combination:
    def __init__(
            self,
            all_tokens,
            library,
            name: str,
            type=None,
            id=0,
            prefix_expression=None,
            phy_units=None,
            description: str = None,
    ):
        '''token type and representation'''
        token_type_err_msg = f"token_type must be one of ['combination']"
        assert type in ['combination'], token_type_err_msg
        self.token_type = type
        self.token_type_kind_number = TYPE_KIND_NUMBER[type] if type in TYPE_KIND_NUMBER else -1
        self.representation = name
        self.token_name = name
        self.infix_expression = name
        self.token_func = None

        '''token idx and attributes'''
        self.legal = True
        self.token_id = id
        self.token_arity = 0
        self.dimension_analysis_case = -1
        self.all_tokens_id_dict = library.all_tokens_id_dict

        '''tokens_idx and token_list'''
        self.tokens_idx = []
        self.prefix_expression = prefix_expression
        for token in prefix_expression:
            if token in self.all_tokens_id_dict:  # legal
                self.tokens_idx.append(self.all_tokens_id_dict[token])
            elif token.isdigit() and int(token) >= 3:  # "2","3"
                for need_add in range(int(token) - 2):
                    self.tokens_idx.append(self.all_tokens_id_dict["add"])
                    self.tokens_idx.append(self.all_tokens_id_dict["1"])
                self.tokens_idx.append(self.all_tokens_id_dict["1"])
            elif token not in self.all_tokens_id_dict:  # illegal
                self.legal = False
        self.tokens_list = [all_tokens[idx] for idx in self.tokens_idx]

        '''complexity and prog'''
        self.length = len(self.tokens_idx)
        complexity = [library.all_token_info_table.complexity_table[idx] for idx in self.tokens_idx]
        self.complexity = sum(complexity)
        self.combination_prog = SRprogram(
            all_tokens_info=None,
            tokens=self.tokens_list,
            free_const_values=None,
            fixed_const_values=None,
        )

        '''phy_units'''
        units = np.full((UNITS_VECTOR_SIZE), 0.)
        units[:len(phy_units)] = np.array(phy_units)
        self.phy_units = units  # (UNITS_VECTOR_SIZE,) of float
        self.is_constraining_phy_units = False if np.isnan(self.phy_units[0]) else True
        self.is_power = False
        self.power = np.nan
        self.get_height(env_library=library)
        phy_units_result = self.get_units_from_bottom(env_library=library)
        if len(phy_units_result) > 0:
            self.phy_units = phy_units_result[0]
        else:
            self.phy_units = None  # have no token in combination, so illegal
        self.description = description

    def __repr__(self):
        return self.representation

    def update_prog_relation(self, env_library):
        library = copy.deepcopy(env_library)
        library._reset()
        for i in range(len(self.combination_prog.tokens)):
            new_tokens_idx = np.full((env_library.batch_size),
                                     fill_value=self.all_tokens_id_dict[self.combination_prog.tokens[
                                         len(self.combination_prog.tokens) - 1].token_name],
                                     # last token is not op (self.all_tokens_id_dict["end"] not in dict)
                                     dtype=int)  # or it will IndexError: index 60 is out of bounds for axis 1 with size 60, because arity2 op
            new_tokens_idx[0] = self.combination_prog.tokens[i].token_id  # our token_id
            update_library(library, new_tokens_idx=new_tokens_idx)
        library.get_height()
        library.get_children_end_and_length()
        # self.library.super_parent_info.total_complexity += library.total_complexity[0]
        # self.library.super_parent_info.parent_index[0,:] = copy.deepcopy(library.parent_index[0, :])
        # self.library.super_parent_info.sibling_index[0,:] = copy.deepcopy(library.sibling_index[0, :])
        self.height = copy.deepcopy(library.height[0, :self.length])
        self.parent_index = copy.deepcopy(library.parent_index[0, :self.length])
        self.sibling_index = copy.deepcopy(library.sibling_index[0, :self.length])
        self.children_index = copy.deepcopy(library.children_index[0, :self.length])
        self.children_number = copy.deepcopy(library.children_number[0, :self.length])
        self.children_end = copy.deepcopy(library.children_end[0, :self.length])

    def get_height(self, env_library: SRlibrary):
        self.height = np.zeros((1, self.length))
        self.parent_index = np.full((1, self.length), fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.sibling_index = np.full((1, self.length), fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.children_index = np.full((1, self.length, 2), fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.children_number = np.zeros((1, self.length))
        self.children_end = np.full((1, self.length, 2), fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.update_prog_relation(env_library)

    def get_units_from_bottom(self, env_library: SRlibrary):
        arity = env_library.all_token_info_table.arity_table[self.tokens_idx]
        dimension_analysis_case = env_library.all_token_info_table.dimension_analysis_case_table[self.tokens_idx]
        phy_units_table = env_library.all_token_info_table.phy_units_table
        phy_units_table = phy_units_table.reshape([-1, UNITS_VECTOR_SIZE])
        phy_units = phy_units_table[self.tokens_idx]
        end_index = self.length - 1
        # Error messages
        error_msg_unknown_dim = "Unknown physical units token encountered in bottom up units assignment " \
                                "process."
        error_msg_dimensionless_child = "Non-dimensionless token encountered as child of dimensionless op (eg cos, " \
                                        "exp, log etc) in bottom up units assignment process."
        error_msg_dimensionless_token = "Dimensionless token having non-zero physical units encountered in bottom " \
                                        "up units assignment process."
        error_msg_additive_discrepancy = "Two children of binary_additive_op (eg: addition, subtraction) having " \
                                         "different physical units encountered in bottom up units assignment process."
        error_msg_incomplete_tree = "Regular bottom up dimensional analysis can not be performed on incomplete " \
                                    "tree (containing terminal tokens with unknown physical units: eg. dummies)"

        def parser(index):
            if index == -1:
                return True
            parent_idx = self.tokens_idx[index]
            if arity[index] == 0:
                pass
            elif arity[index] == 1:
                # Position of the lonely child of the token (arity = 1)
                child_index = self.children_index[index][0]
                if child_index >= self.length:  # IndexError: index 11 is out of bounds for axis 0 with size 11
                    self.legal = False
                    return False
                # child_idx = self.tokens_idx[child_index]
                child_phy_units = phy_units[child_index]
                if dimension_analysis_case[index] == 3:
                    n_power = env_library.all_token_info_table.power_table[parent_idx]
                    phy_units[index, :] = n_power * child_phy_units
                # Elif token is an unary additive op -> copy-paste units from child
                elif dimension_analysis_case[index] == 4:  # same as child
                    phy_units[index, :] = child_phy_units
                # Elif token is an unary dimensionless op -> nothing to do but making sure that child token is
                # dimensionless (as it should be) just in case and that current token is dimensionless
                elif dimension_analysis_case[index] == 5:  # need to be dimensionless
                    # assert np.array_equal(child_phy_units, np.zeros(7)), error_msg_dimensionless_child
                    if not np.array_equal(child_phy_units, np.zeros(7)):
                        self.legal = False
                        return False
                    phy_units[index, :] = np.zeros(7)
            elif arity[index] == 2:
                # Children positions
                child0_index = self.children_index[index][0]
                child1_index = self.children_index[index][1]
                if child0_index >= self.length or child1_index >= self.length:  # IndexError: index 11 is out of bounds for axis 0 with size 11
                    self.legal = False
                    return False
                # Child 0 units
                child0_phy_units = phy_units[child0_index]
                # Child 1 units
                child1_phy_units = phy_units[child1_index]
                # If token is an additive token -> units are those of any children (as they should be the same
                # among them) but making sure that children of additive binary tokens have the same units for safety.
                if dimension_analysis_case[index] == 1:
                    # assert np.array_equal(child1_phy_units, child0_phy_units), error_msg_additive_discrepancy
                    if not np.array_equal(child1_phy_units, child0_phy_units):
                        self.legal = False
                        return False
                    phy_units[index, :] = child0_phy_units
                # Elif token is a multiplicative token
                elif dimension_analysis_case[index] == 20 or dimension_analysis_case[index] == 21:
                    # token = child0 * child1 => units(token) = child0_phy_units + child1_phy_units
                    if dimension_analysis_case[index] == 20:
                        phy_units[index, :] = child0_phy_units + child1_phy_units
                    # token = child0 / child1 => units(token) = child0_phy_units - child1_phy_units
                    elif dimension_analysis_case[index] == 21:
                        phy_units[index, :] = child0_phy_units - child1_phy_units
            parser(index - 1)

        parser(end_index)
        return phy_units
