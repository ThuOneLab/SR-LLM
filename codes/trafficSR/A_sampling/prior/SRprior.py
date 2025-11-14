import numpy as np
from abc import ABC
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import TYPE_KIND_NUMBER
from codes.trafficSR.A_sampling.env_composition.SRlibrary import ILLEGAL_RELATIVE_INDEX

DEFAULT_INVERSE_OPERATOR = {
    # "add": "sub",
    # "sub": "add",
    # "mul": "div",
    # "div": "mul",
    "exp": "log",
    "log": "exp",
    "inv": "inv",
    "neg": "neg",
    "n2": "sqrt",
    # "n3": "sqrt",
    "n4": "sqrt",
    "sqrt": "n2",
    "arctan" : "tan",
    "tan"    : "arctan",
    "arcsin" : "sin",
    "sin"    : "arcsin",
    "arccos" : "cos",
    "cos"    : "arccos",
}

DEFAULT_OFFSET_OPERATOR = ["sub", "div"]

# should kid have no these operator
DEFAULT_TRANSCENDENTAL_OPERATOR = [
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    # "n2",
    # "n3",
    # "n4",
    # "inv",
    # "sqrt",
    # "pow",
    "abs",
]

DEFAULT_RISKY_OPERATOR = [
    "div",
    "inv"
]

# TRIGONOMETRIC OPS
TRIGONOMETRIC_OP = ["sin", "cos", "tan", "tanh", "sinh", "cosh", "arctan", "arccos", "arcsin"]
POWER_OP = ["pow", "n4", "n3"] # 过于高维度，不适合嵌套
DEFAULT_NEST_OPERATOR={
    "exp":["exp"],
    "log":["log"],
    "trig":TRIGONOMETRIC_OP,
    "pow":POWER_OP, #["pow"],
    "abs":["abs"],
}
fourOperators=['add', 'sub', 'mul', 'div', 'inv', 'neg', 'n2', 'sqrt']
class Prior(ABC):
    def __init__(
            self,
            env,
    ):
        self.env = env
        self.all_tokens = env.all_tokens
        self.batch_size = env.batch_size
        self.max_time_step = env.max_time_step
        self.tokens_number = env.tokens_number
        self._reset_prior()

    def _reset_prior(self):
        self.prior_matrix = np.zeros((self.batch_size, self.env.tokens_number))

    def __call__(self):
        return self.prior_matrix


# correct
class HardLengthPrior(Prior):
    '''
    to ensure the length of the program is within the range of [min_length, max_length]
    '''

    def __init__(
            self,
            env,
            min_length=None,
            max_length=None,
    ):
        Prior.__init__(
            self,
            env,
        )
        if min_length is None:
            min_length = 3
        if max_length is None:
            max_length = env.max_time_step
        assert min_length >= 1, "min_length must be greater than 1!"
        assert max_length <= env.max_time_step, "max_length must be less than max_time_step!"
        self.min_length = min_length
        self.max_length = max_length
        all_tokens_id = np.arange(0, self.tokens_number)
        self.mask_arity_0_no_combination = np.isin(all_tokens_id, np.array(
            self.env.library.all_token_info_table.arity_0_no_combination_group))
        
        self.arity0= np.array(self.env.library.all_token_info_table.arity_0_group)
        self.mask_arity_0 = np.isin(all_tokens_id, self.arity0)
        
        self.arity1 = np.array(self.env.library.all_token_info_table.arity_1_group)
        self.mask_arity_1 = np.isin(all_tokens_id, self.arity1)
        
        self.arity2 = np.array(self.env.library.all_token_info_table.arity_2_group)
        self.mask_arity_2 = np.isin(all_tokens_id, self.arity2)
        
        # 长度（包括combination）
        self.token_length = np.array(self.env.library.all_token_info_table.length_table)
        # self.arity_0_no_combination_group = self.env.library.all_token_info_table.arity_0_no_combination_group

    def __call__(
            self,
            progs,
    ):
        self._reset_prior() # self.prior_matrix = np.zeros((self.batch_size, self.env.tokens_number))
        completed_length = progs.completed_length
        real_length = self.env.library.real_length
        
        '''first token 需要是operator'''
        if completed_length == 0:
            self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
            self.prior_matrix[:, self.env.library.all_tokens_id_dict['end']] = 0
            if self.min_length > 1:
                self.prior_matrix[:, self.env.library.all_token_info_table.arity_0_group] = 0
            return self.prior_matrix

        '''shorter than min length'''
        prior_matrix_min_length = np.ones((self.batch_size, self.env.tokens_number))
        
        # real_length如果<min_length-1，则need to be longer: 更不可能choose arity_0_no_combination token
        mask_real_length_less_min_length = real_length < self.min_length - 1
        mask_one_placeholder = (progs.library.n_placeholder_now == 1)  # Progs having only one dummy (going to finish at next step if choosing a terminal token)
        
        # 哪些progs既小于min_length，又只有一个placeholder
        mask_need_longer = np.logical_and(mask_real_length_less_min_length, mask_one_placeholder)
        
        # 哪些token需要被禁止：arity==0且长度不达标
        # next_token_cannot_use_too_short = np.outer(mask_need_longer, self.mask_arity_0_no_combination)
        # prior_matrix_min_length[next_token_cannot_use_too_short] = 0
        min_length_arity0=self.min_length-real_length #如果取arity0，则最小长度是
        invalid_length=self.token_length[None, :] < min_length_arity0[:, None]
        invalid_arity0 = np.logical_and(invalid_length, self.mask_arity_0)
        next_token_cannot_use_too_short = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False, dtype=bool)
        for i in np.where(mask_need_longer)[0]:
            next_token_cannot_use_too_short[i, invalid_arity0[i]] = True
        prior_matrix_min_length[next_token_cannot_use_too_short] = 0 #将arity0且长度不达标的token置为概率0
        
        '''longer than max length'''
        next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False, dtype=bool)
        n_placeholders_now = self.env.library.n_placeholder_now # 添加new_token前，目前的placeholder数量
        '''已经completed，缺乏placeholder，就选择end'''
        # sum_placeholder<=0, which means it is completed
        mask_case = n_placeholders_now <= 0
        next_token_bool[mask_case, np.array(
            self.env.library.all_token_info_table.arity_end_group).reshape(
            (-1, 1))] = True
        
        '''还未completed，跟max_length进行比较，发现其需要哪些字符——这里会把end排除掉'''
        mask_haveplaceholders = n_placeholders_now > 0
        # 已经考虑了combination的情况
        # 总结规律：(0)这里面arity1和arity2一定是长度为1的符号。（1）目前还剩下self.max_time_step - self.env.library.real_length个空位，记为left_length。left_length-n_placeholders_now记为n_redundancy冗余，用来判断能加入arity哪些符号。（2）left_length-n_placeholders_now+1，是当前格取arity_0时候的最大长度max_length_arity0。（3）如果left_length-n_placeholders_now==0，那么只能取arity_0。（4）如果left_length-n_placeholders_now==1，那么只能取arity_0和arity_1。（5）如果left_length-n_placeholders_now>=2，那么可以取arity_0和arity_1和arity_2，否则只能取arity_0和arity_1。
        left_length=self.max_time_step - self.env.library.real_length
        n_redundancy=left_length-n_placeholders_now
        max_length_arity0=n_redundancy+1
        valid_arity0 = self.token_length[None, :] <= max_length_arity0[:, None]
        valid_arity0 = np.logical_and(valid_arity0, self.mask_arity_0) #这是用来指示能取哪些不超过长度的arity0符号,维度是（batchsize,n_tokens）
        
        # case1:只能取arity0
        mask_case1 = np.logical_and(mask_haveplaceholders, n_redundancy == 0)
        for i in np.where(mask_case1)[0]:
            next_token_bool[i, valid_arity0[i]] = True
        
        # case2:只能取arity0和arity1
        mask_case2 = np.logical_and(mask_haveplaceholders, n_redundancy == 1)
        for i in np.where(mask_case2)[0]:
            next_token_bool[i, valid_arity0[i]] = True
            next_token_bool[i, self.mask_arity_1] = True
            
        # case3:取arity0和arity1和arity2
        mask_case2 = np.logical_and(mask_haveplaceholders, n_redundancy >= 2)
        for i in np.where(mask_case2)[0]:
            next_token_bool[i, valid_arity0[i]] = True
            next_token_bool[i, self.mask_arity_1] = True
            next_token_bool[i, self.mask_arity_2] = True
        
        '''# real_length!!!:left length for current token 
        left_length = self.max_time_step - (
                self.env.library.real_length + n_placeholders_now)
        # case1:only one placeholder left
        mask_case = np.logical_and(mask_haveplaceholders, left_length == 1)
        next_token_bool[mask_case, np.array(self.arity_0_no_combination_group).reshape((-1, 1))] = True
        # case2:two placeholders left
        mask_case = np.logical_and(mask_haveplaceholders, left_length == 2)
        next_token_bool[mask_case, np.array(
            self.env.library.all_token_info_table.arity_0_group + self.env.library.all_token_info_table.arity_1_group).reshape(
            (-1, 1))] = True
        # case3:more than 3 placeholders left
        mask_case = np.logical_and(mask_haveplaceholders, left_length >= 3)
        next_token_bool[mask_case, np.array(
            self.env.library.all_token_info_table.arity_0_group + self.env.library.all_token_info_table.arity_1_group + self.env.library.all_token_info_table.arity_2_group).reshape(
            (-1, 1))] = True
        # case4:should choose shorter or equal length: also consider combination to be False
        mask_case_longer = np.tile(np.array(self.env.library.all_token_info_table.length_table),
                                   (self.batch_size, 1)) > left_length.reshape(-1,
                                                                               1)
        next_token_bool[mask_case_longer] = False'''

        # --------------------------update prior_matrix
        self.prior_matrix[next_token_bool] = True
        self.prior_matrix = np.multiply(self.prior_matrix, prior_matrix_min_length)
        return self.prior_matrix

# combination
class SoftLengthPrior(Prior):
    '''
    to encourage the length of the program to be close to the optimal length
    '''

    def __init__(
            self,
            env,
            length_loc=None,
            scale=None,
            eps=1e-2,
    ):
        Prior.__init__(
            self,
            env,
        )
        try:
            length_loc = float(length_loc)
        except ValueError:
            " length_loc must be cast-able to a float"
        try:
            scale = float(scale)
        except ValueError:
            " scale must be cast-able to a float"
        # If we want length = 3, gaussian value must be max at step = 2 (ie. when generating token n°3)
        self.length_loc = int(length_loc)  # 2.0  / 1.7
        # => step_loc = length_loc - 1
        self.step_loc = float(self.length_loc) - 1
        self.scale = float(scale)
        # Value of gaussian at all steps
        steps = np.arange(0, self.max_time_step + 1)  # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
        self.gaussian_vals = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale))
        # self.gaussian_vals = np.where(self.gaussian_vals >= eps, self.gaussian_vals, eps)
        # Is token of the library a terminal token : mask
        all_tokens_id = np.arange(0, self.tokens_number)
        self.mask_arity_0_no_combination = np.isin(all_tokens_id,
                                                   np.array(
                                                       self.env.library.all_token_info_table.arity_0_no_combination_group))
        self.mask_operator = np.isin(all_tokens_id, np.array(self.env.library.all_token_info_table.operator_group))

    def __call__(
            self,
            progs,
    ):
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        # about combination group, need to be considered
        # current_time_step = self.env.library.current_time_step
        real_length = self.env.library.real_length
        real_length_gaussian_vals = self.gaussian_vals[real_length]
        real_length_gaussian_vals = np.expand_dims(real_length_gaussian_vals, 1).repeat(self.env.tokens_number, axis=1)

        # Before loc: need to be longer to meet optimal length_loc: more likely to not choose
        # Scale terminal token probs by gaussian where progs have only one dummy
        mask_before_loc = real_length < self.step_loc
        mask_one_placeholder_progs = (
                progs.library.n_placeholder_now == 1)  # Progs having only one dummy (going to finish at next step if choosing a terminal token) : mask
        mask_need_longer = np.logical_and(mask_one_placeholder_progs, mask_before_loc)
        mask_should_longer = np.outer(mask_need_longer, self.mask_arity_0_no_combination) # TODO：先这样，后续考虑combination
        self.prior_matrix[mask_should_longer] *= real_length_gaussian_vals[
            mask_should_longer]  # shorter than step_loc: less probability to choose arity-0 tokens
        '''
        # At loc: gaussian value = 1.
        # After loc: need to stop sooner; Scale non-terminal tokens probs by gaussian
        mask_after_loc = real_length > self.step_loc
        # mask_operator = np.logical_not(self.mask_arity_0)
        # mask_operator[-1] = False  # 'end' need to be 1
        mask_should_stop = np.outer(mask_after_loc, self.mask_operator)
        self.prior_matrix[mask_should_stop] *= real_length_gaussian_vals[
            mask_should_stop]  # longer than step_loc: less probability to choose operator tokens
        '''
        return self.prior_matrix

# 用于替换SoftLengthPrior
class SoftMaxLengthPrior(Prior):
    '''
    to encourage the length of the program to be close to the optimal length(only when >step_loc， it will work)
    '''

    def __init__(
            self,
            env,
            length_loc=None,
            scale=None,
            eps=1e-2,
    ):
        Prior.__init__(
            self,
            env,
        )
        try:
            length_loc = float(length_loc)
        except ValueError:
            " length_loc must be cast-able to a float"
        try:
            scale = float(scale)
        except ValueError:
            " scale must be cast-able to a float"
        self.length_loc = int(length_loc)  # 2.0  / 1.3
        # => step_loc = length_loc - 1
        self.step_loc = float(self.length_loc) - 1 #这就是想要开始给出最大长度软限制的地方
        self.scale = float(scale)
        # Value of gaussian at all steps
        steps = np.arange(0,
                          self.max_time_step + 1)  # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
        self.gaussian_vals = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale))
        self.gaussian_vals = np.where(self.gaussian_vals >= eps, self.gaussian_vals, eps)

        all_tokens_id = np.arange(0, self.tokens_number)
        self.mask_operator = np.isin(all_tokens_id, np.array(self.env.library.all_token_info_table.operator_group))

    def __call__(
            self,
            progs,
    ):
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        # about combination group, need to be considered
        # current_time_step = self.env.library.current_time_step
        real_length = self.env.library.real_length
        real_length_gaussian_vals = self.gaussian_vals[real_length]
        real_length_gaussian_vals = np.expand_dims(real_length_gaussian_vals, 1).repeat(self.env.tokens_number, axis=1)

        # At loc: gaussian value = 1.
        # After loc: need to stop sooner, so don't use operator; Scale non-terminal tokens probs by gaussian
        mask_after_loc = real_length > self.step_loc
        mask_should_stop = np.outer(mask_after_loc, self.mask_operator) # 不让它再加operator
        self.prior_matrix[mask_should_stop] *= real_length_gaussian_vals[
            mask_should_stop]  # less probability to choose operator tokens
        return self.prior_matrix


# correct: equal to max length judge
class ArityPrior(Prior):
    '''
    to ensure programs are valid, satisfying the arity constraints
    '''

    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )

    def __call__(
            self,
            progs
    ):
        self._reset_prior()
        completed_length = progs.completed_length

        if completed_length == 0:
            self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
            self.prior_matrix[:, -1] = 0
            return self.prior_matrix

        sum_placeholders = self.env.library.n_placeholder_now
        next_token_prior = []
        for prog_id in range(self.batch_size):
            if sum_placeholders[prog_id] <= 0:
                next_token_prior.append(self.env.library.all_token_info_table.arity_end_group)
            else:
                min_completed_length = completed_length + sum_placeholders[prog_id] + 1
                if self.max_time_step - min_completed_length == 0:
                    next_token_prior.append(self.env.library.all_token_info_table.arity_0_group)
                elif self.max_time_step - min_completed_length == 1:
                    next_token_prior.append(
                        self.env.library.all_token_info_table.arity_0_group + self.env.library.all_token_info_table.arity_1_group)
                elif self.max_time_step - min_completed_length >= 2:
                    next_token_prior.append(
                        self.env.library.all_token_info_table.arity_0_group + self.env.library.all_token_info_table.arity_1_group + self.env.library.all_token_info_table.arity_2_group)
                else:
                    raise ValueError(f"the {prog_id}th of the programs has a valid length over max_time_step!")

        for prog_id in range(self.batch_size):
            self.prior_matrix[prog_id, next_token_prior[prog_id]] = 1

        return self.prior_matrix

# 只保留关于1op的。并且这个约束现在no use
# correct:const can not be child of 1-op AND end can not be child of operator; 
# not use:fixed const can not be left child of 2-op
class ConstPrior(Prior):
    '''
    if the last token is an operator, the next token can be any token except for the const token
    '''

    def __init__(
            self,
            env
    ):
        Prior.__init__(
            self,
            env,
        )

    def __call__(
            self,
            progs
    ):
        self._reset_prior()
        next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=True, dtype=bool) #（batchsize,n_tokens）用于获知哪些True token可以被选取
        if progs.completed_length > 0:  # completed_length=0: pass
            current_tokens_idx = progs.library.tokens_idx
            last_token_arity = self.env.library.all_token_info_table.arity_table[
                current_tokens_idx[:, progs.completed_length - 1]]

            # judge last token is operator, const can not be child of 1-op AND end can not be child of operator
            mask_arity_1 = last_token_arity == 1
            next_token_bool[mask_arity_1, np.array(
                self.env.library.all_token_info_table.free_const_group + self.env.library.all_token_info_table.semi_free_const_group +
                self.env.library.all_token_info_table.fixed_const_group + self.env.library.all_token_info_table.end_group).reshape(
                (-1, 1))] = False
            # judge last token is operator, fixed const can not be left child of 2-op： avoid add 1 1;
            # don't use: and free/semi_free can not be left child of 2-op: avoid sub alpha alpha
            # mask_arity_2 = last_token_arity == 2
            # next_token_bool[mask_arity_2, np.array(
            #     self.env.library.all_token_info_table.end_group).reshape(
            #     (-1, 1))] = False #self.env.library.all_token_info_table.fixed_const_group + 
        # The feasible prior element is 1, otherwise it is 0
        self.prior_matrix[next_token_bool] = 1
        return self.prior_matrix


# correct, no use
class ConstOncePrior(Prior):
    '''
    free_const token can only be chosen once in each program
    '''

    def __init__(
            self,
            env
    ):
        Prior.__init__(
            self,
            env,
        )
        self.have_picked_free_const = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False, dtype=bool)

    def __call__(
            self,
            progs
    ):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        if progs.completed_length > 0:  # completed_length=0: pass
            current_tokens_idx = progs.library.tokens_idx
            last_token_idx = current_tokens_idx[:, progs.completed_length - 1]
            last_token_type = self.env.library.all_token_info_table.type_table[last_token_idx]
            # TODO: add combination description
            mask_type_free_const = last_token_type == TYPE_KIND_NUMBER[
                'free_const']  # judge last token is free const, if true, then add in have_picked_free_const: can not show again
            batch_id = np.arange(start=0, stop=self.batch_size, step=1)
            free_const_coords = np.stack((batch_id[mask_type_free_const], last_token_idx[mask_type_free_const]), axis=0)
            self.have_picked_free_const[tuple(free_const_coords)] = True
            # self.have_picked_free_const[
            #     mask_type_free_const, last_token_idx[mask_type_free_const].reshape((-1, 1))] = True # error: it will make have_picked_free_const[mask_type_free_const,[3,4,5]]= True
        # The feasible prior element is 1, otherwise it is 0
        self.prior_matrix[self.have_picked_free_const] = 0
        return self.prior_matrix


# correct
class NoneSingleArityInversePrior(Prior):
    '''
    to prevent single-arity operators' child is their opposite operator
    '''

    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )
        self.inverse_operator_keys = np.array(list(DEFAULT_INVERSE_OPERATOR.keys()))
        self.inverse_operator_values = np.array(list(DEFAULT_INVERSE_OPERATOR.values()))

    def __call__(
            self,
            progs
    ):
        self._reset_prior()
        completed_length = progs.completed_length
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        if completed_length == 0:
            return self.prior_matrix
        '''version 1
        # last_tokens = current_tokens_idx[:, completed_length - 1]
        # for prog_id in range(self.batch_size):
        #     last_token = self.all_tokens[last_tokens[prog_id]]
        #     last_token_name = last_token.token_name
        #
        #     if last_token.token_arity == 1 and last_token.token_name in DEFAULT_INVERSE_OPERATOR:
        #         inverse_operator_name = DEFAULT_INVERSE_OPERATOR[last_token_name]
        #         if inverse_operator_name not in self.env.library.all_tokens_id_dict:
        #             continue
        #         inverse_operator_id = self.env.library.all_tokens_id_dict[inverse_operator_name]
        #         self.prior_matrix[prog_id, inverse_operator_id] = 0
        '''
        '''version 2
        # next_token_inverse_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False,dtype=bool)  # True means inverse token id
        # if progs.completed_length > 0:  # completed_length=0: pass
        #     current_tokens_idx = progs.library.tokens_idx
        #     last_token_arity = self.env.library.all_token_info_table.arity_table[
        #         current_tokens_idx[:, progs.completed_length - 1]]
        #     last_token_names = self.env.library.all_token_info_table.token_name_table[
        #         current_tokens_idx[:, progs.completed_length - 1]]
        #     mask_arity_1 = last_token_arity == 1
        #     mask_name_in_inverse_operator = np.isin(last_token_names, self.inverse_operator_keys)
        #
        #     last_token_inverse = np.full(shape=self.batch_size, fill_value='', dtype='<U32')
        #     mask_have_1op_inverse = np.logical_and(mask_arity_1, mask_name_in_inverse_operator)
        #     last_token_inverse[mask_have_1op_inverse] = np.array(
        #         [DEFAULT_INVERSE_OPERATOR[last_token_name] for last_token_name in
        #          last_token_names[mask_have_1op_inverse]])
        #     mask_last_token_inverse_in_all_tokens = np.isin(last_token_inverse,
        #                                                     self.env.library.all_token_info_table.token_name_table)
        #     last_token_inverse_id = np.full(shape=self.batch_size, fill_value=-1, dtype=int)
        #     last_token_inverse_id[mask_last_token_inverse_in_all_tokens] = np.array(
        #         [self.env.library.all_tokens_id_dict[last_token_name] for last_token_name in
        #          last_token_inverse[mask_last_token_inverse_in_all_tokens]])
        #     # inverse can not be child of operator
        #     next_token_inverse_bool[mask_last_token_inverse_in_all_tokens, last_token_inverse_id[
        #         mask_last_token_inverse_in_all_tokens].reshape(
        #         (-1, 1))] = True
        '''
        
        next_token_inverse_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False,dtype=bool)  # True means inverse token id
        if progs.completed_length > 0:  # completed_length=0: pass
            last_tokens_idx = progs.library.tokens_idx[:, progs.completed_length - 1]
            last_token_arity = self.env.library.all_token_info_table.arity_table[last_tokens_idx]
            inverse_last_tokens_idx = progs.library.all_token_info_table.inverse_token_id_table[last_tokens_idx]
            mask_arity_1 = last_token_arity == 1
            mask_have_inverse = inverse_last_tokens_idx != ILLEGAL_RELATIVE_INDEX
            mask_1op_and_have_inverse = np.logical_and(mask_arity_1, mask_have_inverse)
            batch_id = np.arange(start=0, stop=self.batch_size, step=1)
            inverse_coords = np.stack(
                (batch_id[mask_1op_and_have_inverse], inverse_last_tokens_idx[mask_1op_and_have_inverse]), axis=0)
            next_token_inverse_bool[tuple(inverse_coords)] = True
        self.prior_matrix[next_token_inverse_bool] = 0
        return self.prior_matrix
# TODO:sub,x_1,add,x_1,c_3

# correct
class NoneDoubleArityInversePrior(Prior):
    '''
    1. Inverse operator token can only be chosen as the right child of the parent operator token
    2. The left child of the parent operator token can not be same as the right child of the inverse operator token

    '''

    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )
        self.inverse_operator_keys = np.array(list(DEFAULT_INVERSE_OPERATOR.keys()))
        self.inverse_operator_values = np.array(list(DEFAULT_INVERSE_OPERATOR.values()))

    def __call__(
            self,
            progs
    ):
        self._reset_prior()
        completed_length = progs.completed_length
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        if completed_length == 0:
            return self.prior_matrix
        # for prog_id in range(self.batch_size):
        #     last_token = self.all_tokens[last_tokens[prog_id]]
        #     last_token_name = last_token.token_name
        #
        #     # if last token is an operator and its arity is 2, the next cannot be its inverse operator
        #     if last_token_name in DEFAULT_INVERSE_OPERATOR and last_token.token_arity == 2:
        #         inverse_operator_name = DEFAULT_INVERSE_OPERATOR[last_token_name]
        #         if inverse_operator_name not in self.env.library.all_tokens_id_dict:
        #             continue
        #         inverse_operator_id = self.env.library.all_tokens_id_dict[inverse_operator_name]
        #         self.prior_matrix[prog_id, inverse_operator_id] = 0
        #         continue

        next_token_inverse_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=False, dtype=bool)  # True means inverse token id
        # if progs.completed_length > 0:  # completed_length=0: pass
        #     current_tokens_idx = progs.library.tokens_idx
        #     last_token_arity = self.env.library.all_token_info_table.arity_table[
        #         current_tokens_idx[:, progs.completed_length - 1]]
        #     last_token_names = self.env.library.all_token_info_table.token_name_table[
        #         current_tokens_idx[:, progs.completed_length - 1]]
        #     mask_arity_2 = last_token_arity == 2
        #     mask_name_in_inverse_operator = np.isin(last_token_names, self.inverse_operator_keys)
        #     mask_2op_and_have_inverse = np.logical_and(mask_arity_2, mask_name_in_inverse_operator)
        #
        #     last_token_inverse = np.full(shape=self.batch_size, fill_value='', dtype='<U32')
        #     last_token_inverse[mask_2op_and_have_inverse] = np.array(
        #         [DEFAULT_INVERSE_OPERATOR[last_token_name] for last_token_name in
        #          last_token_names[mask_2op_and_have_inverse]])
        #     mask_last_token_inverse_in_all_tokens = np.isin(last_token_inverse,
        #                                                     self.env.library.all_token_info_table.token_name_table)
        #     last_token_inverse_id = np.full(shape=self.batch_size, fill_value=-1, dtype=int)
        #     last_token_inverse_id[mask_last_token_inverse_in_all_tokens] = np.array(
        #         [self.env.library.all_tokens_id_dict[last_token_name] for last_token_name in
        #          last_token_inverse[mask_last_token_inverse_in_all_tokens]])
        #     # inverse can not be child of operator
        #     next_token_inverse_bool[mask_last_token_inverse_in_all_tokens, last_token_inverse_id[
        #         mask_last_token_inverse_in_all_tokens].reshape(
        #         (-1, 1))] = True
        # # The feasible prior element is 1, otherwise it is 0
        # self.prior_matrix[next_token_inverse_bool] = 0

        if progs.completed_length > 0:  # completed_length=0: pass
            last_tokens_idx = progs.library.tokens_idx[:, progs.completed_length - 1]
            last_token_arity = self.env.library.all_token_info_table.arity_table[last_tokens_idx]
            inverse_last_tokens_idx = progs.library.all_token_info_table.inverse_token_id_table[last_tokens_idx] # 最新填写的符号的反符号
            mask_arity_2 = last_token_arity == 2
            mask_have_inverse = inverse_last_tokens_idx != ILLEGAL_RELATIVE_INDEX
            mask_2op_and_have_inverse = np.logical_and(mask_arity_2, mask_have_inverse) # 说明之前填写的一个符号，是2op and have inverse
            batch_id = np.arange(start=0, stop=self.batch_size, step=1)
            inverse_coords = np.stack(
                (batch_id[mask_2op_and_have_inverse], inverse_last_tokens_idx[mask_2op_and_have_inverse]), axis=0)
            next_token_inverse_bool[tuple(inverse_coords)] = True
        self.prior_matrix[next_token_inverse_bool] = 0
        return self.prior_matrix


# correct
class NoneDoubleArityOffsetPrior(Prior):
    '''
    1. avoid sub 1 1 or div alpha alpha
    '''

    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )

    def __call__(self, progs):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))

        def mask_to_coords(mask):
            self.pos = np.tile(np.arange(0, progs.completed_length), (self.batch_size, 1)).astype(int)
            self.pos_batch = np.tile(np.arange(0, self.batch_size), (progs.completed_length, 1)).transpose().astype(
                int)
            mask_sum = mask.sum()  # int
            coords = np.zeros(shape=(2, mask_sum,), dtype=int)  # (2, mask_sum,) of int
            coords[:, :] = np.stack((  # (2, mask_sum,) of int
                self.pos_batch[mask],  # batch dim coord
                self.pos[mask],  # time dim coord
            ), axis=0)
            return mask_sum, coords

        completed_length = progs.completed_length
        if completed_length >= 2:  # maybe sub 1 (can not be 1), so have =
            batch_id = np.arange(start=0, stop=self.batch_size, step=1)
            current_tokens_idx = progs.library.tokens_idx[:, :progs.completed_length]

            # get offset op's position
            mask_offset_operator = np.full(shape=(self.batch_size, completed_length), fill_value=False,
                                           dtype=bool)
            for operator in DEFAULT_OFFSET_OPERATOR:
                if operator in self.env.library.all_tokens_id_dict:
                    mask_offset_operator = np.logical_or(mask_offset_operator,
                                                         current_tokens_idx == self.env.library.all_tokens_id_dict[operator])

            # get mask_left_child_completed
            children_index = progs.library.children_index[:, :progs.completed_length]
            right_children_index = children_index[:, :, 1]
            right_children_coords = np.stack(
                (np.tile(batch_id, (completed_length, 1)).transpose(), right_children_index), axis=0)
            mask_left_child_completed = right_children_coords[1] <= completed_length

            mask_offset_left_child_completed = np.logical_and(mask_offset_operator, mask_left_child_completed)
            mask_sum, parent_coords = mask_to_coords(mask_offset_left_child_completed)
            left_child_length = children_index[tuple(parent_coords)][:, 1] - children_index[tuple(parent_coords)][:, 0]
            for i in range(parent_coords.shape[1]):
                right_children_pos = right_children_index[parent_coords[0, i], parent_coords[1, i]]
                right_child_length = completed_length - right_children_pos
                if right_child_length == left_child_length[i] - 1:
                    parent_children = progs.library.children_index[parent_coords[0, i], parent_coords[1, i]]
                    parent_left_child_idxs = progs.library.tokens_idx[parent_coords[0, i],
                                             parent_children[0]:parent_children[1]]
                    parent_right_child_idxs = progs.library.tokens_idx[parent_coords[0, i],
                                              right_children_pos:completed_length]
                    if np.array_equal(parent_left_child_idxs[:-1], parent_right_child_idxs):
                        self.prior_matrix[parent_coords[0, i], parent_left_child_idxs[-1]] = 0

        return self.prior_matrix


# correct
class NoneDoubleAritySymmetryPrior(Prior):
    '''
    avoid such as + a - b a
    '''

    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )
        self.inverse_operator_keys = np.array(list(DEFAULT_INVERSE_OPERATOR.keys()))
        self.inverse_operator_values = np.array(list(DEFAULT_INVERSE_OPERATOR.values()))

    def __call__(self, progs):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        '''version2'''
        def mask_to_coords(mask):
            self.pos = np.tile(np.arange(0, progs.completed_length), (self.batch_size, 1)).astype(int)
            self.pos_batch = np.tile(np.arange(0, self.batch_size), (progs.completed_length, 1)).transpose().astype(int)
            mask_sum = mask.sum()  # int
            coords = np.zeros(shape=(2, mask_sum,), dtype=int)  # (2, mask_sum,) of int
            coords[:, :] = np.stack((  # (2, mask_sum,) of int
                self.pos_batch[mask],  # batch dim coord
                self.pos[mask],  # time dim coord
            ), axis=0)
            return mask_sum, coords

        completed_length = progs.completed_length
        if completed_length > 3:
            batch_id = np.arange(start=0, stop=self.batch_size, step=1)
            current_tokens_idx = progs.library.tokens_idx[:, :progs.completed_length]
            inverse_tokens_idx = progs.library.all_token_info_table.inverse_token_id_table[current_tokens_idx]
            children_index = progs.library.children_index[:, :progs.completed_length]
            right_children_index = children_index[:, :, 1]
            right_children_coords = np.stack(
                (np.tile(batch_id, (completed_length, 1)).transpose(), right_children_index), axis=0)
            mask_right_child_exist = right_children_coords[1] < completed_length
            mask_left_child_completed = right_children_coords[1] <= completed_length

            mask_sum, have_right_child_coords = mask_to_coords(mask_right_child_exist)
            mask_right_children_left_children_completed = np.full(shape=(self.batch_size, completed_length),
                                                                  fill_value=False,
                                                                  dtype=bool)
            right_child_right_child_index = np.stack(
                (have_right_child_coords[0], right_children_index[tuple(have_right_child_coords)]), axis=0)
            mask_right_children_left_children_completed[tuple(have_right_child_coords)] = mask_left_child_completed[
                tuple(right_child_right_child_index)]

            right_children_idx = np.full(shape=(self.batch_size, completed_length), fill_value=-1,
                                         dtype=int)
            right_children_idx[mask_right_children_left_children_completed] = progs.library.tokens_idx[
                tuple(right_children_coords[:, mask_right_children_left_children_completed])]
            mask_right_children_equal_inverse = np.full(shape=(self.batch_size, completed_length), fill_value=False,
                                                        dtype=bool)
            mask_right_children_equal_inverse[:, :completed_length] = right_children_idx == inverse_tokens_idx

            mask_sum, parent_coords = mask_to_coords(mask_right_children_equal_inverse)
            left_child_length = children_index[tuple(parent_coords)][:, 1] - children_index[tuple(parent_coords)][:, 0]
            for i in range(parent_coords.shape[1]):
                right_children_pos = right_children_index[parent_coords[0, i], parent_coords[1, i]]
                right_child_right_child_pos = right_children_index[parent_coords[0, i], right_children_pos]
                right_child_right_child_length = completed_length - right_child_right_child_pos
                if right_child_right_child_length == left_child_length[i] - 1:
                    parent_children = progs.library.children_index[parent_coords[0, i], parent_coords[1, i]]
                    parent_left_child_idxs = progs.library.tokens_idx[parent_coords[0, i],
                                             parent_children[0]:parent_children[1]]
                    right_child_right_child_idxs = progs.library.tokens_idx[parent_coords[0, i],
                                                   right_child_right_child_pos:completed_length]
                    if np.array_equal(parent_left_child_idxs[:-1], right_child_right_child_idxs):
                        self.prior_matrix[parent_coords[0, i], parent_left_child_idxs[-1]] = 0
        '''version1
        # last_token_arity = self.env.library.all_token_info_table.arity_table[
        #     current_tokens_idx[:, 0:completed_length]]
        # last_token_names = self.env.library.all_token_info_table.token_name_table[
        #     current_tokens_idx[:, 0:completed_length]]
        # mask_arity_2 = last_token_arity == 2
        # mask_name_in_inverse_operator = np.isin(last_token_names, self.inverse_operator_keys)
        # mask_2op_and_have_inverse = np.logical_and(mask_arity_2, mask_name_in_inverse_operator)
        #
        # if completed_length > 3:  # completed_length=0: pass
        #     for prog_id in range(self.batch_size):
        #         if mask_2op_and_have_inverse[prog_id, completed_length - 1]:
        #             continue
        #         # The left child of the parent operator token can not be same as the right child of the inverse operator token
        #         # pos at most need to be completed-4
        #         for pos in range(completed_length - 3):  # pos is the position of testing token
        #             parent = self.all_tokens[current_tokens_idx[prog_id, pos]]
        #             # if parent has 2 children AND parent is inverse_opeartor
        #             children_index = self.env.library.children_index[prog_id, pos]
        #             # 2 children are valid
        #             if children_index[1] < completed_length - 1 and mask_2op_and_have_inverse[prog_id, pos]:
        #                 parent_left_child_idxs = current_tokens_idx[prog_id, children_index[0]:children_index[1]]
        #                 parent_left_child_length = children_index[1] - children_index[
        #                     0]  # right child index-left child index
        #                 parent_right_child = self.all_tokens[current_tokens_idx[prog_id, children_index[1]]]
        #
        #                 right_child_right_child_index = self.env.library.children_index[prog_id, children_index[1]][1]
        #                 if parent_right_child.token_name == DEFAULT_INVERSE_OPERATOR[
        #                     parent.token_name] and right_child_right_child_index <= completed_length:  # right child can at [completed_length]
        #                     right_child_right_child_idxs = current_tokens_idx[prog_id,
        #                                                    right_child_right_child_index:completed_length]
        #                     right_child_right_child_length = completed_length - right_child_right_child_index
        #                     if right_child_right_child_length == parent_left_child_length - 1 and \
        #                             np.array_equal(parent_left_child_idxs[:-1], right_child_right_child_idxs):
        #                         self.prior_matrix[prog_id, parent_left_child_idxs[-1]] = 0
        '''

        return self.prior_matrix

# not correct
class NoneTranscendNestPrior(Prior):
    '''
    to prevent transcendental operators from nesting (even in left or right kid)
    '''

    def __init__(
            self,
            env
    ):
        Prior.__init__(
            self,
            env,
        )
        # self.transcend_operator_keys = np.array(DEFAULT_TRANSCENDENTAL_OPERATOR)

    def __call__(
            self,
            progs
    ):
        '''version4:简化版本'''
        self._reset_prior()
        next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=True, dtype=bool)
        transcendental_group=self.env.library.all_token_info_table.transcendental_group
        if len(transcendental_group) > 0 and progs.completed_length > 0:  # completed_length=0: pass
            # 使用progs.library.parent_index判断current position的父节点是否为transcendental operator
            parent_indices = progs.library.parent_index[:, progs.completed_length]#得到当前位置的父节点index
            end_token_idx=self.tokens_number-1
            parent_token_idx=np.array([progs.library.tokens_idx[prog_id, parent_indices[prog_id]] if parent_indices[prog_id]!=ILLEGAL_RELATIVE_INDEX else end_token_idx for prog_id in range(self.batch_size)]) #得到当前位置的父节点token_idx
            mask_can_not_nest = np.isin(parent_token_idx, transcendental_group)# 判定父节点的idx是否在transcendental_group中
            next_token_bool[mask_can_not_nest, np.array(transcendental_group).reshape(-1, 1)] = False# False代表current position不能放置transcendental operator
        self.prior_matrix[next_token_bool] = 1
        
        '''version3: about left kid and right kid
        # self._reset_prior()
        # next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=True, dtype=bool)
        # if len(self.env.library.all_token_info_table.transcendental_group) > 0 and progs.completed_length > 0:  # completed_length=0: pass
        #     # mask_have_children_current_time_step代表有哪些已完成位置，目前它的孩子包括当前时间步
        #     current_time_step = progs.library.current_time_step
        #     past_children_index = progs.library.children_index[:, :progs.completed_length]
        #     mask_have_children_is_current_time_step = np.any(past_children_index == current_time_step, axis=2).reshape(
        #         [self.batch_size, -1])
        #     # mask_in_transcendental代表有哪些位置是transcendental的
        #     past_token_name = self.env.library.all_token_info_table.token_name_table[
        #         progs.library.tokens_idx[:, :progs.completed_length]]
        #     mask_in_transcendental = np.isin(past_token_name, self.transcend_operator_keys)
        #     # 确定哪些transcendental的孩子在当前时间步。则当前时间步不能再放置transcendental符号
        #     mask_can_not_nest = np.logical_and(mask_have_children_is_current_time_step, mask_in_transcendental)
        #     mask_can_not_nest = np.any(mask_can_not_nest, axis=1).reshape([self.batch_size])
        #     next_token_bool[mask_can_not_nest, np.array(
        #         self.env.library.all_token_info_table.transcendental_group).reshape(
        #         (-1, 1))] = False
        # self.prior_matrix[next_token_bool] = 1
        '''
        '''version2
        # self._reset_prior()
        # next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=True, dtype=bool)
        # if len(self.env.library.all_token_info_table.transcendental_group) > 0 and progs.completed_length > 0:  # completed_length=0: pass
        #     current_tokens_idx = progs.library.tokens_idx
        #     last_token_idx = current_tokens_idx[:, progs.completed_length - 1]
        #     last_token_name = self.env.library.all_token_info_table.token_name_table[last_token_idx]
        #     mask_in_transcendental = np.isin(last_token_name, self.transcend_operator_keys)
        #     next_token_bool[mask_in_transcendental, np.array(
        #         self.env.library.all_token_info_table.transcendental_group).reshape(
        #         (-1, 1))] = False
        # self.prior_matrix[next_token_bool] = 1
        '''
        '''version1
        # tokens_name = self.env.library.all_tokens_id_dict.keys()
        # for prog_id in range(self.batch_size):
        #     last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 1]]
        #     if last_token.token_name in DEFAULT_TRANSCENDENTAL_OPERATOR:
        #         for token in DEFAULT_TRANSCENDENTAL_OPERATOR:  # Transcendental ops, should only be 1 depth
        #             if token in tokens_name:
        #                 self.prior_matrix[prog_id, self.env.library.all_tokens_id_dict[token]] = 0
        '''
        return self.prior_matrix


# correct, but no use
class NoneFreeConstRiskyPrior(Prior):
    '''
    to ensure free const token is not the right child of the arity-two risky operator, 
    or the child of arity-one risky operator
    '''

    def __init__(
            self,
            env
    ):

        Prior.__init__(
            self,
            env,
        )

    def __call__(
            self,
            progs,
    ):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        current_tokens_idx = progs.library.tokens_idx
        completed_length = progs.completed_length

        next_token_prior = []
        prior_matrix_1 = np.zeros((self.batch_size, self.env.tokens_number))

        if completed_length == 0:
            return self.prior_matrix

        # arity-one risky operator
        for prog_id in range(self.batch_size):
            last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 1]]
            if last_token.token_name in DEFAULT_RISKY_OPERATOR and last_token.token_arity == 1:
                next_token_prior.append(
                    self.env.library.all_token_info_table.operator_group + self.env.library.all_token_info_table.variable_group + self.env.library.all_token_info_table.fixed_const_group + self.env.library.all_token_info_table.end_group)
            else:
                next_token_prior.append(
                    self.env.library.all_token_info_table.operator_group + self.env.library.all_token_info_table.variable_group + self.env.library.all_token_info_table.free_const_group + self.env.library.all_token_info_table.fixed_const_group + self.env.library.all_token_info_table.end_group)

        for prog_id in range(self.batch_size):
            prior_matrix_1[prog_id, next_token_prior[prog_id]] = 1

        # arity-two risky operator
        prior_matrix_2 = np.ones((self.batch_size, self.env.tokens_number))
        for prog_id in range(self.batch_size):
            for pos in range(completed_length):
                tok = self.all_tokens[current_tokens_idx[prog_id, pos]]
                if tok.token_arity == 2 and tok.token_name in DEFAULT_RISKY_OPERATOR:
                    left_child_sum_arity = 1
                    left_child_done = False
                    for pos_after in range(pos + 1, completed_length):
                        left_child_sum_arity += self.all_tokens[current_tokens_idx[prog_id, pos_after]].token_arity - 1
                        if left_child_sum_arity == 0:
                            left_child_done = True
                            break
                    if not left_child_done:
                        break
                    else:
                        if completed_length - pos_after == 1:
                            prior_matrix_2[prog_id, self.env.library.all_token_info_table.free_const_group] = 0
                        else:
                            break
        self.prior_matrix = np.multiply(prior_matrix_1, prior_matrix_2)
        return self.prior_matrix


class PhysicalUnitsPrior(Prior):
    """
    Enforces that next token should be physically consistent units-wise with current program based on current units
    constraints computed live (during program generation). If there is no way get a constraint all tokens are allowed.
    """

    def __init__(self, env, prob_eps=None):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        prob_eps : float
            Value to return for the prior inplace of zeros (useful for avoiding sampling problems)
        """
        # ------- INITIALIZING -------
        Prior.__init__(self, env)
        # Tolerance when comparing two units vectors (eg. 0.333333334 == 0.333333333)
        self.tol = 1e2 * np.finfo(np.float32).eps
        # Value to return for the prior inplace of zeros.
        if prob_eps is None:
            self.prob_eps = np.finfo(np.float32).eps
        else:
            self.prob_eps = prob_eps

        # ------- LIB_IS_CONSTRAINING -------
        # mask : are tokens in the library constraining units-wise
        self.lib_is_constraining = self.env.library.all_token_info_table.is_constraining_phy_units_table  # (n_choices,)
        # mask : are tokens in the library constraining units-wise (expanding in a new batch_size axis)
        self.lib_is_constraining_padded = np.tile(self.lib_is_constraining,
                                                  reps=(self.batch_size, 1))  # (batch_size, n_choices,)

        # ------- LIB_UNITS -------
        # Units of choosable tokens in the library
        self.lib_units = self.env.library.all_token_info_table.phy_units_table  # (n_choices, UNITS_VECTOR_SIZE,)
        # Padded units of choosable tokens in the library (expanding in a new batch_size axis)
        self.lib_units_padded = np.tile(self.lib_units,
                                        reps=(self.batch_size, 1, 1))  # (batch_size, n_choices, UNITS_VECTOR_SIZE,)

    def __call__(self, progs):
        # Current step
        curr_step = self.env.library.current_time_step
        # print("*-----------------------------------*")
        # print("physical unit prior-time step:", curr_step)
        # ------- COMPUTE REQUIRED UNITS -------
        # Updating programs with newest most constraining units constraints
        progs.progs_assign_required_units()

        # ------- IS_PHYSICAL -------
        # mask : is dummy at current step part of a physical program units-wise
        is_physical = self.env.library.is_physical  # (batch_size,)
        # mask : is dummy at current step part of a physical program units-wise (expanding in a new n_choices axis)
        is_physical_padded = np.moveaxis(np.tile(is_physical, reps=(self.tokens_number, 1))  # (batch_size, n_choices,)
                                         , source=0, destination=1)

        # ------- IS_CONSTRAINING -------
        # mask : does dummy at current step contain constraints units-wise
        is_constraining = self.env.library.is_constraining_phy_units[:, curr_step]  # (batch_size,)
        # mask : does dummy at current step contain constraints units-wise (expanding in a new n_choices axis)
        is_constraining_padded = np.moveaxis(np.tile(is_constraining, reps=(self.tokens_number, 1))
                                             # (batch_size, n_choices,)
                                             , source=0, destination=1)
        # Number of programs in batch that constraining at this step
        n_constraining = is_constraining.sum()
        # mask : for each token in batch, for each token in library are both tokens constraining
        mask_prob_is_constraining_info = np.logical_and(self.lib_is_constraining_padded,
                                                        is_constraining_padded)  # (batch_size, n_choices,)

        # Useful as to forbid a choice, the choosable token must be constraining and the current dummy must also be
        # constraining, otherwise the choice should be legal regardless of the units of any of these tokens
        # (non-constraining tokens should contain NaNs units).

        # ------- UNITS -------
        # Units requirements at current step dummies
        units_requirement = self.env.library.phy_units[:, curr_step, :]  # (batch_size, UNITS_VECTOR_SIZE)
        # Padded units requirements of dummies at current step (expanding in a new n_choices axis)
        units_requirement_padded = np.moveaxis(np.tile(units_requirement, reps=(self.tokens_number, 1, 1))
                                               # (batch_size, n_choices, UNITS_VECTOR_SIZE)
                                               , source=0, destination=1)
        # mask : for each token in batch, is choosing token in library legal units-wise
        mask_prob_units_legality = (np.abs(units_requirement_padded - self.lib_units_padded) < self.tol).prod(
            axis=-1)  # (batch_size, n_choices)

        # ------- RESULT -------
        # Token in library should be allowed if there are no units constraints on any side (library, current dummies)
        # OR if the units are consistent OR if the program is unphysical.
        # Ie. all tokens in the library are allowed if there are no constraints on any sides or if the program is
        # unphysical anyway.
        mask_prob = np.logical_or.reduce((  # (batch_size, n_choices)
            (~ mask_prob_is_constraining_info),
            (~ is_physical_padded),
            mask_prob_units_legality,
        )).astype(float)
        mask_prob[mask_prob == 0] = self.prob_eps
        return mask_prob

# correct:2op不能两个孩子都为const
class NoneDoubleConstPrior(Prior):
    '''
    to ensure the childs of arity-two operator token are not both const token
    '''

    def __init__(
            self,
            env
    ):

        Prior.__init__(
            self,
            env,
        )

    def __call__(
            self,
            progs,
    ):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        current_tokens_idx = progs.library.tokens_idx
        completed_length = progs.completed_length
        
        '''version2'''
        next_token_cannot_be_const = (
                    self.env.library.all_token_info_table.free_const_group
                    + self.env.library.all_token_info_table.semi_free_const_group
                    + self.env.library.all_token_info_table.fixed_const_group
                )
        if completed_length == 0 or completed_length == 1:
            return self.prior_matrix
        # arity-one risky operator
        for prog_id in range(self.batch_size):
            last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 1]]
            last_last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 2]]
            if (
                    last_token.token_type == "free_const" or last_token.token_type == "semi_free_const" or last_token.token_type == "fixed_const") \
                    and (last_last_token.token_type == "operator" and last_last_token.token_arity == 2):
                self.prior_matrix[prog_id, next_token_cannot_be_const] = 0

        '''version1
        # next_token_prior = []
        # if completed_length == 0 or completed_length == 1:
        #     return self.prior_matrix
        # # arity-one risky operator
        # for prog_id in range(self.batch_size):
        #     last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 1]]
        #     last_last_token = self.all_tokens[current_tokens_idx[prog_id, completed_length - 2]]
        #     if (
        #             last_token.token_type == "free_const" or last_token.token_type == "semi_free_const" or last_token.token_type == "fixed_const") \
        #             and (last_last_token.token_type == "operator" and last_last_token.token_arity == 2):
        #         next_token_prior.append(
        #             self.env.library.all_token_info_table.free_const_group +
        #             self.env.library.all_token_info_table.semi_free_const_group +
        #             self.env.library.all_token_info_table.fixed_const_group)
        #     else:
        #         next_token_prior.append([])

        # for prog_id in range(self.batch_size):
        #     if len(next_token_prior[prog_id]) > 0:
        #         self.prior_matrix[prog_id, next_token_prior[prog_id]] = 0
        '''
        return self.prior_matrix

# power第一个和第二个位置的填入规则
class PowerPrior(Prior):
    """
    让Power的第一个孩子为变量，第二个孩子为常数（或变量），根据child2_mode确定——默认为常数
    """

    def __init__(
        self,
        env,
        # child2_mode="const"
    ):
        Prior.__init__(
            self,
            env,
        )
        # self.child2_mode = child2_mode

    def __call__(self, progs):
        """version1"""
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        completed_length = progs.completed_length
        if completed_length == 0:
            return self.prior_matrix

        current_tokens_idx = progs.library.tokens_idx
        next_token_need_be_variable = self.env.library.all_token_info_table.variable_group # self.env.library.all_token_info_table.operator_group+
        next_token_need_be_const = (
                    self.env.library.all_token_info_table.operator_group
                    + self.env.library.all_token_info_table.free_const_group
                    + self.env.library.all_token_info_table.semi_free_const_group
                    # + self.env.library.all_token_info_table.fixed_const_group
                )

        # arity-one risky operator
        for prog_id in range(self.batch_size):
            last_token = self.all_tokens[
                current_tokens_idx[prog_id, completed_length - 1]
            ]
            last_last_token = (
                self.all_tokens[current_tokens_idx[prog_id, completed_length - 2]]
                if completed_length > 1
                else None
            )
            if last_token.token_name == "pow":
                self.prior_matrix[prog_id, :] = 0
                self.prior_matrix[prog_id, next_token_need_be_variable] = 1
            elif last_last_token is not None and last_last_token.token_name == "pow" and last_token.token_type == "variable":
                self.prior_matrix[prog_id, :] = 0
                self.prior_matrix[prog_id, next_token_need_be_const] = 1

        return self.prior_matrix
    
# correct，but worse
class FirstNotFourOperatorsPrior(Prior):
    '''
    to ensure the length of the program is within the range of [min_length, max_length]
    '''

    def __init__(
            self,
            env,
            min_length,
            NotFour=False,
    ):
        Prior.__init__(
            self,
            env,
        )
        self.min_length = min_length
        self.NotFour = NotFour
        
        self.mask_four_operators = np.isin(self.env.library.all_token_info_table.token_name_table, fourOperators) # 如add sub mul div
        
        all_tokens_id = np.arange(self.env.tokens_number)
        self.operator_group= np.array(self.env.library.all_token_info_table.operator_group)
        self.mask_operators = np.isin(all_tokens_id, self.operator_group)
        
        self.mask_not_four_operators = ~self.mask_four_operators & self.mask_operators # 如exp pow log abs等
    def __call__(
            self,
            progs,
    ):
        completed_length = progs.completed_length
        
        '''first token'''
        if completed_length == 0:
            if self.min_length > 1:
                self.prior_matrix = np.zeros((self.batch_size, self.env.tokens_number))
                if self.NotFour:
                    self.prior_matrix[:, self.mask_not_four_operators] = 1
                else:
                    self.prior_matrix[:, self.mask_four_operators] = 1
            return self.prior_matrix
        else:
            return np.ones((self.batch_size, self.env.tokens_number))


class NoneNestedPrior(Prior):
    '''
    to prevent nested operators
    '''
    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )
        combination_group_idx = self.env.library.all_token_info_table.combination_group
        combination_group_prefix=[self.env.all_tokens[combination_idx].prefix_expression for combination_idx in combination_group_idx]
        self.nested_idx=[]
        for i in DEFAULT_NEST_OPERATOR.keys():
            nestI=[]
            token_name_list=DEFAULT_NEST_OPERATOR[i]
            for token in token_name_list:
                if token in self.env.library.all_tokens_id_dict:
                    nestI.append(self.env.library.all_tokens_id_dict[token])
                for j,combination_prefix in enumerate(combination_group_prefix):
                    if token in combination_prefix: # 如果嵌套符号出现在某个combination_prefix中
                        nestI.append(combination_group_idx[j]) # 则将combination_idx加入
            if len(nestI)>0:
                self.nested_idx.append(nestI)
        self.max_nesting_depth=1 # 最大嵌套深度 不可以sin(cos)

    def __call__(
            self,
            progs,
    ):
        self._reset_prior()
        self.prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        completed_length = progs.completed_length
        if completed_length == 0 or len(self.nested_idx)==0:
            return self.prior_matrix

        for prog_id in range(self.batch_size):
            ancestors_idx = progs.library.get_ancestor_idx_of_progi(prog_id, completed_length) #  - 1是bug，因为应该取当前位置的祖先。否则SRprogram : (['mul', 'sub', 'x_1', 'mul', 'x_1', 'exp', 'sqrt', 'inv', 'add', 'exp', 'exp', 'neg', 'sub', '1', 'add', 'c_3', 'sin', 'neg', 'x_1', 'c_4', 'c_2'])
            if len(ancestors_idx) > 0: # 存在祖先
                for i in range(len(self.nested_idx)): #分析每种嵌套
                    nest_count=0
                    for nidx in self.nested_idx[i]: # 分析每种嵌套中的每个禁止token nidx:不会包含combination，因为combination arity=0
                        nest_count += ancestors_idx.count(nidx) # 统计祖先中禁止token的数量
                    if nest_count >= self.max_nesting_depth: #改成大于等于
                        self.prior_matrix[prog_id, self.nested_idx[i]] = 0 # 这里包括了combination
                        # if token in ancestors_idx: # 如果祖先中存在禁止token
                        #     self.prior_matrix[prog_id, self.nested_idx[i]] = 0
                        #     break
        return self.prior_matrix

# 当存在sincos时使用，但是发现worse
class SinCosConstPrior(Prior):
    '''
    to prevent nested operators
    '''
    def __init__(
            self,
            env,
    ):
        Prior.__init__(
            self,
            env,
        )
        self.const_forbidden_idx = []
        for token in ["sin","cos"]:
            if token in self.env.library.all_tokens_id_dict:
                self.const_forbidden_idx.append(self.env.library.all_tokens_id_dict[token])

    def __call__(
            self,
            progs,
    ):
        self._reset_prior()
        next_token_bool = np.full(shape=(self.batch_size, self.tokens_number), fill_value=True, dtype=bool) #（batchsize,n_tokens）用于获知哪些True token可以被选取
        if progs.completed_length > 0:  # completed_length=0: pass
            last_token_idx = self.env.library.tokens_idx[:, progs.completed_length - 1]
            # judge last token is in sin/cos, const can not be left child of sin/cos
            mask_sin_cos = np.isin(last_token_idx, self.const_forbidden_idx)
            next_token_bool[mask_sin_cos, np.array(
                self.env.library.all_token_info_table.free_const_group + self.env.library.all_token_info_table.semi_free_const_group +
                self.env.library.all_token_info_table.fixed_const_group + self.env.library.all_token_info_table.end_group).reshape(
                (-1, 1))] = False
        self.prior_matrix[next_token_bool] = 1
        return self.prior_matrix

DEFAULT_PRIOR_TYPE = [
    "HardLength",
    "SoftLength",
    "SoftMaxLength",
    'Arity',
    'Const',
    "ConstOnce",
    "NoneSingleArityInverse",
    "NoneDoubleArityInverse",
    "NoneDoubleArityOffset",
    "NoneDoubleAritySymmetry",
    "NoneTranscendNest",
    "NoneFreeConstRisky",
    "PhysicalUnits",
    "NoneDoubleConst",
    "Power",
    "FirstNotFourOperators",
    "NoneNested",
    "SinCosConst",
]

class Prior_Collector():
    def __init__(
            self,
            env,
            prior_args
    ):
        self.env = env
        self.batch_size = env.batch_size
        self.prior_types = prior_args["prior_type"]
        self.prior_config = prior_args["prior_config"]
        self.HardLengthPrior = HardLengthPrior(env, self.prior_config["LengthPrior"]["min_length"], self.prior_config["LengthPrior"]["max_length"])
        self.SoftLengthPrior = SoftLengthPrior(env, self.prior_config["SoftLengthPrior"]["length_loc"], self.prior_config["SoftLengthPrior"]["scale"], self.prior_config["SoftLengthPrior"]["eps"])
        self.SoftMaxLengthPrior = SoftMaxLengthPrior(env, self.prior_config["SoftLengthPrior"]["max_length_loc"], self.prior_config["SoftLengthPrior"]["scale"], self.prior_config["SoftLengthPrior"]["eps"])
        self.ConstPrior = ConstPrior(env)
        self.ConstOncePrior = ConstOncePrior(env)
        self.NoneSingleArityInversePrior = NoneSingleArityInversePrior(env)
        self.NoneDoubleArityInversePrior = NoneDoubleArityInversePrior(env)
        self.NoneDoubleArityOffsetPrior = NoneDoubleArityOffsetPrior(env)
        self.NoneDoubleAritySymmetryPrior = NoneDoubleAritySymmetryPrior(env)
        self.NoneTranscendNestPrior = NoneTranscendNestPrior(env)
        self.PhysicalUnitsPrior = PhysicalUnitsPrior(env, self.prior_config["PhysicalUnitsPrior"]["prob_eps"])
        self.NoneDoubleConstPrior = NoneDoubleConstPrior(env)
        self.PowerPrior = PowerPrior(env)
        self.FirstNotFourOperatorsPrior = FirstNotFourOperatorsPrior(env, self.prior_config["LengthPrior"]["min_length"], self.prior_config["FirstNotFourOperatorsPrior"]["NotFour"])
        self.NoneNestedPrior = NoneNestedPrior(env)
        self.SinCosConstPrior = SinCosConstPrior(env)

    def __call__(
            self,
    ):
        prior_matrix = np.ones((self.batch_size, self.env.tokens_number))
        for prior_type in self.prior_types:
            assert prior_type in DEFAULT_PRIOR_TYPE, f"unknown prior type {prior_type}!"
            prior_matrix = np.multiply(prior_matrix, eval(f"self.{prior_type}Prior")(self.env.programs))
            pass

        return prior_matrix # 发现pow x1 2（如果第一个取了pow，则由于pow变量常量约束，直接说明长度要为3）和min_length=4是冲突的
