import codes.trafficSR.A_sampling.env_tokens_combinations.SRfunctions as SRfunctions
import numpy as np
import torch

# Number of units in SI system [m, s, kg, K, A, cd, mol], two assistant units: rad Sr
UNITS_VECTOR_SIZE = 7


class operator:
    def __init__(self, arity, function, dimension_analysis_case, complexity=1):
        self.arity = arity
        self.function = function
        self.dimension_analysis_case = dimension_analysis_case
        self.complexity = complexity

# (arity, function, dimension_analysis_case, complexity)
DEFAULT_OPERATOR = {
    "add": operator(2, SRfunctions.add, 1, 2),
    "sub": operator(2, SRfunctions.subtract, 1, 2),

    "mul": operator(2, SRfunctions.multiply, 20, 2),
    "div": operator(2, SRfunctions.protected_div, 21, 2),
    # "pow": operator(2, SRfunctions.protected_pow, 22, 2), # check the dimension_analysis_case——发现应该是5

    "inv": operator(1, SRfunctions.protected_inverse, 3, 3),
    "n2": operator(1, SRfunctions.protected_n2, 3, 3),
    "n3": operator(1, SRfunctions.protected_n3, 3, 3),
    "n4": operator(1, SRfunctions.protected_n4, 3, 3),
    "sqrt": operator(1, SRfunctions.protected_sqrt, 3, 3),

    "abs": operator(1, torch.abs, 4, 1),
    "neg": operator(1, torch.negative, 4, 1),

    "exp": operator(1, SRfunctions.protected_exp, 5, 4),
    "log": operator(1, SRfunctions.protected_log, 5, 4),
    "sin": operator(1, SRfunctions.sin, 5, 4),
    "cos": operator(1, SRfunctions.cos, 5, 4),
    "tan": operator(1, SRfunctions.tan, 5, 4),
    "sinh": operator(1, SRfunctions.sinh, 5, 4),
    "cosh": operator(1, SRfunctions.cosh, 5, 4),
    "tanh": operator(1, SRfunctions.tanh, 5, 4),
    "arcsin": operator(1, SRfunctions.protected_arcsin, 5, 4),
    "arccos": operator(1, SRfunctions.protected_arccos, 5, 4),
    "arctan": operator(1, SRfunctions.arctan, 5, 4),
    
    "pow": operator(2, SRfunctions.protected_pow, 5, 4),
}

# 便于sympy print表示
OPS_REPRESENTATIONS = {
    # unprotected operator
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "n2": "n2",
    "neg": "-",
    "abs": "abs",
    "inv": "1/",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "arctan": "arctan",
    "arccos": "arccos",
    "arcsin": "arcsin",
    "erf": "erf",
    "logabs": "logabs",
    "expneg": "expneg",
    "n3": "n3",
    "n4": "n4",
    "pow": "Pow",
    "max": "max",
}

# POWER VALUES OF POWER TOKENS
OP_POWER_VALUE_DICT = {
    "n2": 2,
    "sqrt": 0.5,
    "n3": 3,
    "n4": 4,
    "inv": -1,
}

UNARY_DIMENSIONLESS_OP = ["sin", "cos", "tan", "exp", "log", "expneg", "logabs", "sigmoid", "tanh", "sinh", "cosh", "harmonic", "arctan", "arccos", "arcsin", "erf", "pow"]

TYPE_KIND_NUMBER = {
    "variable": 0,
    "free_const": 1,
    "fixed_const": 2,
    "operator": 3,
    "combination": 4,
    "end": -1, }

COMMUTATIVE_FUNCTIONS = ["add", "mul", "max", "min"]


class Token(object):
    def __init__(
            self,
            name,
            type,
            id,
            func=None,
            fixed_value=None,
            phy_units=None,
            description=None,
    ):
        self.token_name = name
        token_type_err_msg = f"token_type must be one of ['variable', 'free_const', 'operator', 'fixed_const', 'combination','end']"
        assert type in ['operator', 'variable', 'free_const', 'semi_free_const', 'fixed_const', 'combination', 'end'], token_type_err_msg
        self.token_type = type
        self.token_type_kind_number = TYPE_KIND_NUMBER[type] if type in TYPE_KIND_NUMBER else -1
        self.representation = OPS_REPRESENTATIONS[name] if type == 'operator' else name
        self.token_func = func if func is not None else None
        self.token_id = id
        self.length = 0 if type == "end" else 1  # end need to be 0length, because we calculate real length
        self.prefix_expression = [self.token_name]
        self.infix_expression = self.token_name
        self.description = description
        self.tokens_idx = [self.token_id]
        self.tokens_list = None

        if type == 'operator':
            assert name in DEFAULT_OPERATOR.keys() and type == 'operator', \
                f"unknown operator name {name}"

        if type == 'end':
            assert name == 'end' or name == 'placeholder', \
                f"end token's name must be chosen as 'name'!"

        if type == 'operator':
            self.token_arity = DEFAULT_OPERATOR[name].arity
        elif type == 'end':
            self.token_arity = -1
        else:
            self.token_arity = 0

        self.token_func = DEFAULT_OPERATOR[name].function if type == 'operator' else None
        self.dimension_analysis_case = DEFAULT_OPERATOR[name].dimension_analysis_case if type == 'operator' else -1
        if type == 'operator':
            self.complexity = DEFAULT_OPERATOR[name].complexity
        elif type == 'end':
            self.complexity = 0
        else:
            self.complexity = 1

        if fixed_value is not None:
            assert type == 'fixed_const', f"fixed_value can only be defined for fixed_const token"
        self.fixed_value = fixed_value

        if phy_units is None:  # operator
            if self.token_name in UNARY_DIMENSIONLESS_OP:
                # no list definition in default arg
                self.phy_units = np.full((UNITS_VECTOR_SIZE), 0.)  # (UNITS_VECTOR_SIZE,) of float
            else:  # end or 2-op
                # no list definition in default arg
                self.phy_units = np.full((UNITS_VECTOR_SIZE), np.nan)  # (UNITS_VECTOR_SIZE,) of float
        else:
            # must be a numpy.array to support operations
            units = np.full((UNITS_VECTOR_SIZE), 0.)
            units[:len(phy_units)] = np.array(phy_units)
            self.phy_units = units  # (UNITS_VECTOR_SIZE,) of float
        self.is_constraining_phy_units = False if np.isnan(self.phy_units[0]) else True

        if name in OP_POWER_VALUE_DICT:
            self.is_power = True
            self.power = OP_POWER_VALUE_DICT[name]
        else:
            self.is_power = False
            self.power = np.nan

    def __call__(self,
                 *args):
        assert self.token_type == 'operator', f"token_type must be 'operator' to be callable"
        assert self.token_func is not None, f"token_func must be defined to be callable"
        return self.token_func(*args)

    def __repr__(self):
        return self.token_name


if __name__ == '__main__':
    add_token = Token(
        name='add',
        type='operator',
        id=0,
        func=np.add
    )
    print(add_token(1, 2))
