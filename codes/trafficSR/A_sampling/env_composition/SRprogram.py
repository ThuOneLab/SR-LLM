import sympy as sympy
import torch
from codes.trafficSR.A_sampling.env_composition.time_out_utils import run_with_timeout_multiprocessing
from codes.trafficSR.utils import run_with_timeout_process

import multiprocessing
import queue

import stopit

class SRprogram():
    def __init__(
            self,
            all_tokens_info=None,
            tokens=None,
            free_const_values=None,
            semi_free_const_values=None,
            fixed_const_values=None,
            bool_use_rough_calibration=False,
            subs=None,  # sympy subs
    ):
        import torch
        # assert len(tokens)-np.sum([tok.token_arity for tok in tokens])==1, \
        #     f"Number of tokens must be equal to the sum of their arities plus 1, to fit the tree structure. "

        self.all_tokens_info = all_tokens_info
        self.tokens = tokens
        self.token_length = len(self.tokens)

        self.bool_use_rough_calibration = bool_use_rough_calibration
        self._count_tokens()
        self.include_free_const = True if self.prog_free_const_tokens_number > 0 or (
                self.bool_use_rough_calibration is False and self.prog_semi_free_const_tokens_number > 0) else False

        # need free_const order, so you can correspond name with value easily
        self.free_const_values = free_const_values
        self.semi_free_const_values = semi_free_const_values
        self.fixed_const_values = fixed_const_values

        self._init_total_function()
        if subs is None:
            # subs = {'s': 1, 'v': 1, 'delta_v': 1, 'alpha': 1, 'b': 1, 'v0': 1, 'T': 1, 's0': 1, '1': 1, }
            subs = {'s': torch.tensor(1), 'v': torch.tensor(1), 'delta_v': torch.tensor(1), 'alpha': torch.tensor(1),
                    'b': torch.tensor(1), 'v0': torch.tensor(1), 'T': torch.tensor(1), 's0': torch.tensor(1),
                    '1': torch.tensor(1), }
        self.subs = subs
        self.program_sympy = None

    def __len__(self):
        return len(self.tokens)

    def _init_sympy_function(self, do_simplify=True):
        self.program_sympy = self.get_infix_sympy(do_simplify=do_simplify)

    def _count_tokens(
            self,
    ):
        # self.prog_variable_tokens_number, \
        #     self.prog_fixed_const_tokens_number, \
        #     self.prog_semi_free_const_tokens_number, \
        #     self.prog_free_const_tokens_number, \
        #     self.prog_operator_tokens_number, \
        #     self.prog_combination_tokens_number, \
        #     self.prog_end_tokens_number = 0, 0, 0, 0, 0, 0, 0
        self.prog_semi_free_const_tokens_number, self.prog_free_const_tokens_number = 0, 0
        self.prog_fixed_const, self.prog_semi_free_const, self.prog_free_const = [], [], []
        for tok in self.tokens:
            # if tok.token_type == 'variable':
            #     self.prog_variable_tokens_number += 1
            if tok.token_type == 'fixed_const':
                self.prog_fixed_const.append(tok.token_name)
                # self.prog_fixed_const_tokens_number += 1
            elif tok.token_type == 'semi_free_const':
                self.prog_semi_free_const.append(tok.token_name)
                self.prog_semi_free_const_tokens_number += 1
            elif tok.token_type == 'free_const':
                self.prog_free_const.append(tok.token_name)
                self.prog_free_const_tokens_number += 1
            # elif tok.token_type == 'operator':
            #     self.prog_operator_tokens_number += 1
            # elif tok.token_type == 'combination':
            #     self.prog_combination_tokens_number += 1
            # elif tok.token_type == 'end':
            #     self.prog_end_tokens_number += 1
            # else:
            #     raise ValueError(f"Unknown token type {tok.token_type}")

    def is_completed(
            self,
    ):
        import numpy as np
        return len(self.tokens) - np.sum([tok.token_arity for tok in self.tokens]) == 1

    def _init_total_function(
            self,
    ):
        self.total_function = self.execute

    def execute(
            self,
            X,
            free_const_values=None,
            semi_free_const_values=None
    ):
        X = X.requires_grad_(True)
        free_const_values = free_const_values if free_const_values is not None else self.free_const_values[0]
        semi_free_const_values = semi_free_const_values if semi_free_const_values is not None else \
            self.semi_free_const_values[0]
        n_tokens = len(self.tokens)
        stack = []
        for pos in range(n_tokens - 1, -1, -1):
            token = self.tokens[pos]
            if token.token_arity == 0:
                if token.token_type == 'operator':
                    raise ValueError(f"Operator token {token} must have arity > 0")
                elif token.token_type == 'fixed_const':
                    fixed_const_value = self.fixed_const_values[
                        token.token_id - self.all_tokens_info.operator_tokens_number]
                    tensor = fixed_const_value.repeat(X.shape[0]).to(X.device, X.dtype)
                    stack.append(tensor)
                elif token.token_type == 'semi_free_const':
                    semi_free_const_value = semi_free_const_values[
                        token.token_id - self.all_tokens_info.operator_tokens_number - self.all_tokens_info.fixed_const_tokens_number]
                    tensor = semi_free_const_value.repeat(X.shape[0]).to(X.device, X.dtype)
                    stack.append(tensor)
                elif token.token_type == 'free_const':
                    free_const_value = free_const_values[
                        token.token_id - self.all_tokens_info.operator_tokens_number - self.all_tokens_info.fixed_const_tokens_number- self.all_tokens_info.semi_free_const_tokens_number]
                    tensor = free_const_value.repeat(X.shape[0]).to(X.device, X.dtype)
                    stack.append(tensor)
                elif token.token_type == 'variable':
                    stack.append(X[:, token.token_id-self.all_tokens_info.operator_tokens_number-self.all_tokens_info.fixed_const_tokens_number-self.all_tokens_info.semi_free_const_tokens_number-self.all_tokens_info.free_const_tokens_number])
                else:
                    raise ValueError(f"Unknown token type {token.type}")
            elif token.token_arity > 0:
                operator_values = stack[-token.token_arity:][::-1]
                res = token.token_func(*operator_values)
                stack = stack[:-token.token_arity]
                stack.append(res)
            elif token.token_arity == -1:
                continue

            else:
                raise ValueError(f"Token {token} has err arity")
        # if stack[0].shape == torch.Size([]):
        #     grad_fn = stack[0].grad_fn
        #     stack[0] = torch.full(size=[X.shape[0]], fill_value=float(stack[0])).to(X.device, X.dtype)
        #     stack[0].grad_fn = grad_fn
        #     stack[0].is_leaf = False
        #     stack[0].requires_grad = True
        return stack[0]

    # no use
    def execute_sympy(self, X,
                      free_const_values=None,
                      semi_free_const_values=None):
        import torch

        if self.program_sympy is None:
            self._init_sympy_function()
        X = X.requires_grad_(True)
        free_const_values = free_const_values if free_const_values is not None else self.free_const_values
        semi_free_const_values = semi_free_const_values if semi_free_const_values is not None else self.semi_free_const_values
        dict_keys = list(self.subs.keys())
        for i in range(X.shape[1] + len(free_const_values) + len(semi_free_const_values)):
            if i < X.shape[1]:
                self.subs[dict_keys[i]] = X[0][i]
            elif i >= X.shape[1] and i < X.shape[1] + len(free_const_values):
                self.subs[dict_keys[i]] = free_const_values[i - X.shape[1]]
            else:
                self.subs[dict_keys[i]] = semi_free_const_values[i - X.shape[1] - len(free_const_values)]
        value = self.program_sympy.evalf(
            subs=self.subs)
        if type(value) == sympy.core.numbers.Zero:
            return torch.tensor(0.)
        elif type(value) == sympy.core.numbers.Float:
            return torch.tensor(float(value))
        else:
            return torch.tensor(float(value.args[0]))

    def __call__(self, X):
        return self.total_function(X)

    def get_prior(
            self
    ):
        pass

    def __repr__(self):
        expression = []
        for tok in self.tokens:
            if tok.token_type == 'end':
                break
            expression.append(tok.__repr__())
        return f"SRprogram : ({expression})"

    def get_prefix_expression(self):
        expression = []
        for tok in self.tokens:
            if tok.token_type == 'end':
                break
            expression.append(tok.__repr__())
        return f"{expression}"

    def const_subtitution_representation(
        self, token, free_const_values=None, semi_free_const_values=None
    ):
        free_const_values = free_const_values if free_const_values is not None else self.free_const_values[0] # 因为easybench只有一个数据源，所以index为0代表第一个数据源
        semi_free_const_values = semi_free_const_values if semi_free_const_values is not None else torch.tensor([])
        if token.token_type == "operator":
            raise ValueError(f"Operator token {token} must have arity > 0")
        elif token.token_type == "variable":
            return token.representation
        elif token.token_type == "free_const":
            free_const_value = free_const_values[
                token.token_id - self.all_tokens_info.operator_tokens_number - self.all_tokens_info.fixed_const_tokens_number- self.all_tokens_info.semi_free_const_tokens_number
            ]
            return "{:.2f}".format(free_const_value.cpu().detach().numpy())
        elif token.token_type == "semi_free_const":
            semi_free_const_value = semi_free_const_values[
                token.token_id
                - self.all_tokens_info.operator_tokens_number
                - self.all_tokens_info.fixed_const_tokens_number
            ]
            return "{:.2f}".format(semi_free_const_values.cpu().detach().numpy())
        elif token.token_type == "fixed_const":
            fixed_const_value = self.fixed_const_values[
                token.token_id
                - self.all_tokens_info.operator_tokens_number
            ]
            return "{:.2f}".format(fixed_const_value.cpu().detach().numpy())
        elif token.token_type == "combination": # 因为会在取infix表达式前，将combination替换为tokens，所以这里不会出现combination
            return token.representation

    def get_infix_notation(self,no_const_subtitution=True):
        """
        Computes infix str representation of a program.
        (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
        Parameters
        ----------

        Returns
        -------
        program_str : str
        """
        # Number of tokens in the program
        n_tokens = len(self.tokens)

        # Current stack of computed results
        curr_stack = []

        # De-stacking program (iterating from last token to first)
        start = n_tokens - 1
        res = None # representation of current token
        for i in range(start, -1, -1):
            token = self.tokens[i]
            # Last pending elements are those needed for next computation (in reverse order)
            if token.token_arity == -1:
                continue
            args = curr_stack[-token.token_arity:][::-1]
            if token.token_arity == 0:
                res = token.representation if no_const_subtitution else self.const_subtitution_representation(token)
            elif token.token_arity == 1:
                if token.is_power is True:
                    pow = '{:g}'.format(token.power)  # without trailing zeros
                    res = "((%s)**(%s))" % (args[0], pow)
                    #  if  no_const_subtitution else "((%s)**(%s))" % (self.const_subtitution_representation(args[0]), self.const_subtitution_representation(args[1]))
                else:
                    res = "%s(%s)" % (token.representation, args[0])
            elif token.token_arity == 2:
                if token.token_name == 'pow':
                    res = "((%s)**(%s))" % (args[0], args[1])
                else:
                    res = "(%s%s%s)" % (args[0], token.representation, args[1])
            elif token.token_arity > 2:
                args_str = ""
                for arg in args: args_str += "%s," % arg
                args_str = args_str[:-1]  # deleting last ","
                res = "%s(%s)" % (token.representation, args_str)
            if token.token_arity > 0:
                # Removing those pending elements as they were used
                curr_stack = curr_stack[:-token.token_arity]
            # Appending last result to stack
            curr_stack.append(res)
        return curr_stack[0]

    def simplify_task(self, expr, rational=True):
        return sympy.simplify(expr, rational)

    def get_infix_sympy(self, do_simplify=False, no_const_subtitution=True):
        """
        Returns sympy symbolic representation of a program.
        First get infix_notation, then transform it into sympy.core by pysym.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_sympy : sympy.core
            Sympy symbolic function. It is possible to run program_sympy.evalf(subs={'x': 2.4}) where 'x' is a variable
            appearing in the program to evaluate the function with x = 2.4.
        """
        # import multiprocessing
        # from multiprocessing import Pool
        program_str = self.get_infix_notation(no_const_subtitution=no_const_subtitution)
        program_sympy = sympy.parsing.sympy_parser.parse_expr(program_str, evaluate=False)

        # if do_simplify:
        #     timeout = 20
        #     with Pool(1) as pool:
        #         result = pool.apply_async(self.simplify_task, (program_sympy,))
        #         try:
        #             t0 = time.time()
        #             simplified_expr = result.get(timeout=timeout)
        #             t1 = time.time()
        #             print(f"Function simplify finished in {t1 - t0} seconds")
        #             return simplified_expr
        #         except multiprocessing.context.TimeoutError:
        #             print(f"Function simplify timed out after {timeout} seconds")
        #             return program_sympy
        # else:
        #     return program_sympy
        kwargs={"rational": True}
        if do_simplify:
            '''
            timeout_duration = 5
            func = sympy.simplify
            result_queue = multiprocessing.Queue()
            

            # # 定义实际的工作函数，用于放入新进程中执行
            # def worker(func, args, kwargs, queue):
            #     result = func(*args, **kwargs)
            #     queue.put(result)

            # 创建进程
            p = multiprocessing.Process(target=worker, args=(func, (), {}, result_queue))
            
            # 启动进程
            p.start()
            
            result = None
            try:
                # 等待指定的秒数以获取结果
                result = result_queue.get(timeout=timeout_duration)
            except queue.Empty:
                print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
            finally:
                if p.is_alive():
                    print("Terminating process...")
                    p.terminate()  # 强制终止进程
                    p.join()  # 确保进程已结束
                
            return result if result is not None else program_sympy
            '''
            '''
            program_sympy_simplify = run_with_timeout_process(sympy.simplify, args=(program_sympy,), kwargs={"rational": True},
                                                      timeout_duration=5) #这步太久了
            return program_sympy_simplify
            '''
            timeout=5
            with stopit.ThreadingTimeout(timeout) as tt:
                try:
                    simplify_result = sympy.simplify(program_sympy, rational=False)
                except ZeroDivisionError:
                    print("sympy.simplify encountered ZeroDivisionError; returning unsimplified expression")
                    simplify_result = program_sympy
            if tt.state == tt.EXECUTED:
                # print(f"simplify finished in {timeout}s")
                return simplify_result
            else:
                print(f"simplify timeout after {timeout}s, tt.state={tt.state}")
                return program_sympy

        return program_sympy

    def get_print_expression(self, do_simplify=False):
        """
        Returns a printable ASCII sympy.pretty representation of a program.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_pretty_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify=do_simplify)
        if program_sympy is not None:
            program_pretty_str = sympy.pretty(program_sympy)
            return program_pretty_str
        else:
            return ""

    def get_print_const(self):
        """
        Returns printable const value of a program.
        Parameters
        ----------

        Returns
        -------
        program_free_const_str, program_fixed_const_str : str
        """
        self._count_tokens()
        program_free_const_str = [const + ": " + str(
            self.free_const_values[0].cpu().detach().numpy()[self.all_tokens_info.free_const_tokens_position[const]])
                                  for const in self.prog_free_const]
        program_semi_free_const_str = [const + ": " + str(
            self.semi_free_const_values[0].cpu().detach().numpy()[
                self.all_tokens_info.semi_free_const_tokens_position[const]])
                                       for const in self.prog_semi_free_const]
        program_fixed_const_str = [const + ": " + str(
            self.fixed_const_values.cpu().numpy()[self.all_tokens_info.fixed_const_tokens_position[const]]) for const
                                   in self.prog_fixed_const]
        return program_free_const_str, program_semi_free_const_str, program_fixed_const_str

    def simulate(self, x_lead, v_lead, x_init, v_init, opt_values=None, dt=0.5):
        import torch
        a2_new = torch.zeros(x_lead.shape[0] - 1)
        v2_new = torch.zeros(x_lead.shape[0])
        x2_new = torch.zeros(x_lead.shape[0])

        v2_new[0] = v_init
        x2_new[0] = x_init

        length_x1 = x_lead.shape[0]
        # (x, v, delta_v)
        for k in range(1, length_x1):
            X = torch.tensor([x2_new[k - 1] - x_lead[k - 1], v2_new[k - 1], v2_new[k - 1] - v_lead[k - 1]]).reshape(1,
                                                                                                                    -1)
            if opt_values is not None:
                acc = self.execute(X, opt_values['free_const'], opt_values['semi_free_const'])
            else:
                acc = self.execute(X)
            a2_new[k - 1] = acc
            v2_new[k] = v2_new[k - 1] + acc * dt
            x2_new[k] = x2_new[k - 1] + v2_new[k] * dt

        return a2_new, v2_new, x2_new

    def get_infix_latex(self, replace_dummy_symbol=True, new_dummy_symbol="?", do_simplify=True):
        """
        Returns an str latex representation of a program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_latex_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify=do_simplify)
        program_latex_str = sympy.latex(program_sympy)
        return program_latex_str

def worker(func, args, kwargs, queue):
    """
    工作函数，用于放入新进程中执行。
    :param func: 要执行的目标函数
    :param args: 目标函数的位置参数
    :param kwargs: 目标函数的关键字参数
    :param queue: 用于返回结果的队列
    """
    try:
        result = func(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)
