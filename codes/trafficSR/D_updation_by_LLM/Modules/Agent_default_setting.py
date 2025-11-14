'''
Author: guozelin-ai 3190102461@zju.edu.cn
Date: 2025-02-19 13:55:55
LastEditors: guozelin-ai 3190102461@zju.edu.cn
LastEditTime: 2025-02-19 14:08:56
FilePath: \Symbolic_Regression_with_Large_Language_Models\codes\trafficSR\D_updation_by_LLM\Modules\Agent_default_setting.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
DEFAULT_AGENT_CONFIG = {
    'api_type': 'openai',
    'model_type': 'gpt-4o-mini', #'gpt-4-1106-preview',
    'api_key': 'sk-5hhbGYUx3DEllib085NvhLvjMz5wyRexg4jLepVU4vIXAsRl',
    'temperature': 0.,
    "initial_trial": 1,
    "sleep_time": 5,
}

DEFAULT_TOKEN_DESCRIPTION = {
    "add": "addition",
    "mul": "multiplication",
    "sub": "subtraction",
    "div": "division",
    "n2": "Square",
    "1": "constant 1",
    "k": "sensitivity of driver's acceleration changes",
    "v0": "desired speed of ego",
    "v": "speed of ego",
    "delta_v": "speed difference between ego and the fronted",
    "alpha": "max acceleration of ego",
    "s0": "desired following distance between ego and the fronted",
    "s": "following distance between ego and the fronted",
}

DEFAULT_TOKEN_ARGS = {
    # following are about token_args
    "operator_tokens": ["add", "mul", "sub", "div", "n2", "sqrt"],
    "operator_description": {
        "add": "addition",
        "mul": "multiplication",
        "sub": "subtraction",
        "div": "division",
        "square": "square",
        "sqrt": "square root",
        "pow": "power",
    },
    "variable_tokens": ["s", "v", "delta_v", "a_lead", ],
    "variable_description": {
        "s": "following distance",
        "v": "ego vehicle speed",
        "delta_v": "The speed difference between the front and the ego",
        "a_lead": "acceleration of the leading vehicle",
    },
    "variable_units": {
        "s": [1, 0],
        "v": [1, -1],
        "delta_v": [1, -1],
        "a_lead": [1, -2],
    },
    "fixed_const_tokens": ["1"],
    "fixed_const_values": [1.],
    "fixed_const_units": {"1": [0, 0], },
    "free_const_tokens": ["v0", "s0", "alpha", "b", "T", "a_comfort", "delta", "gamma"],
    "free_const_description": {
        "v0": "desired speed",
        "s0": "desired safe distance",
        "alpha": "comfortable maximum acceleration of the following car",
        "b": "comfortable deceleration level for the driver",
        "T": "desired time headway",
        "a_comfort": "comfortable acceleration level for the driver",
        "delta": "exponent for speed sensitivity",
        "gamma": "exponent for following distance sensitivity",
    },
    "free_const_initial_values": [20., 15., 0.8, 2., 1., 1., 1., 1., 1., 1., 1.],
    "free_const_units": {
        "v0": [1, -1],
        "s0": [1, 0],
        "alpha": [1, -2],
        "b": [1, -2],
        "T": [0, 1],
        "a_comfort": [1, -2],
        "delta": [0, 0],
        "gamma": [0, 0],
    },
    "combination_tokens": [[]],
    "combination_units": {},
    "combination_description": {},
    "combination_prefix_expression": {},
}

delimiter = "####"