new_tokens = [
    {
        "token_name": "T",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "desired time headway",
        "physical_units": {
            "m": 0,
            "s": 1
        }
    },
    {
        "token_name": "a_lead",
        "is_basic": True,
        "token_type": "input_variable",
        "length": 1,
        "description": "acceleration of the leading vehicle",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "a_comfort",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "comfortable acceleration level for the driver",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "delta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "exponent for speed sensitivity",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "Gamma",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "exponent for following distance sensitivity",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "theta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "comfortable maximum acceleration of the following car",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "epsilon",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "coefficient for environmental influence",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "environment",
        "is_basic": True,
        "token_type": "input_variable",
        "length": 1,
        "description": "environmental conditions affecting driving behavior",
        "physical_units": {
            "m": 0,
            "s": 0
        }
    },
    {
        "token_name": "beta",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "parameter for adjusting driver behavior",
        "physical_units": {
            "m": 1,
            "s": -2
        }
    },
    {
        "token_name": "rho",
        "is_basic": True,
        "token_type": "free_const",
        "length": 1,
        "description": "traffic density",
        "physical_units": {
            "m": -1,
            "s": 0
        }
    }
]

'''
def update_TOKEN_ARGS(DEFAULT_TOKEN_ARGS,formula='IDM'):
    if formula == 'IDM' or formula == 'IDM_artifical':
        DEFAULT_TOKEN_ARGS["free_const_tokens"] = ["alpha", "b", "v_0", "T", "s_0"] #之前IDM有"c", "k_1", "k_2"
    # elif formula == 'IDM_artifical': #去掉"c", "k_1", "k_2"
    #     DEFAULT_TOKEN_ARGS["free_const_tokens"] = ["alpha", "b", "v_0", "T", "s_0"]
    elif formula == 'GHR' or formula == 'GHR_artifical':
        DEFAULT_TOKEN_ARGS["variable_tokens"] = ["delta_x", "v", "delta_vi"] #注意这里的s变成了delta_x，delta_v变成了delta_vi
        DEFAULT_TOKEN_ARGS["free_const_tokens"] = ["c"]
        DEFAULT_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"] #直接通用
        # if formula == 'GHR_artifical': #进一步去掉"n2" "sqrt"
        #     DEFAULT_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"]
    elif formula == 'Helly' or formula == 'Helly_artifical':
        DEFAULT_TOKEN_ARGS["variable_tokens"] = ["delta_x", "v", "delta_vi"]
        DEFAULT_TOKEN_ARGS["free_const_tokens"] = ["Beta", "k_1", "k_2"]
        DEFAULT_TOKEN_ARGS["fixed_const_tokens"] = ["1","s_20"]
        DEFAULT_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"] #直接通用
        # if formula == 'Helly_artifical':
        #     DEFAULT_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div"]
    elif formula == 'NEW':
        DEFAULT_TOKEN_ARGS["operator_tokens"] = ["add", "mul", "sub", "div", "n2", "sqrt", "sin", "cos", "exp"]
        DEFAULT_TOKEN_ARGS["variable_tokens"] = ["delta_x", "v", "delta_vi"]
        DEFAULT_TOKEN_ARGS["free_const_tokens"] = ["alpha", "b", "v_0", "T", "s_0"]
        DEFAULT_TOKEN_ARGS["fixed_const_tokens"] = ["1"]
    elif formula=='EASY':
        DEFAULT_TOKEN_ARGS = {"tokens_args": DEFAULT_TOKEN_ARGS,
                "superparent_units": [0,0],
                "superparent_names": ["y"],
                "superparent_prog": [
                ]
        }
    return DEFAULT_TOKEN_ARGS
'''