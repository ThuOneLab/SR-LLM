```python
tokens_info = {
    "operator_tokens": [
        {
            "token_name": "add",
            "token_type": "operator",
            "description": "add"
        },
        {
            "token_name": "mul",
            "token_type": "operator",
            "description": "multiply"
        },
        {
            "token_name": "sub",
            "token_type": "operator",
            "description": "subtract"
        },
        {
            "token_name": "div",
            "token_type": "operator",
            "description": "divide"
        },
        {
            "token_name": "n2",
            "token_type": "operator",
            "description": "square"
        }
    ],
    "variable_tokens": [
        {
            "token_name": "s",
            "token_type": "input_variable",
            "is_basic": True,
            "description": "following distance between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": 0
            },
            "recipes":{
                "variable_tokens":["s"],
                "operator_tokens":[],
                "prefix_expression": ["s"],
            }
        },
        {
            "token_name": "v",
            "token_type": "input_variable",
            "is_basic": True,
            "description": "speed of ego",
            "physical_units": {
                "m": 1,
                "s": -1
            },
            "recipes":{
                "variable_tokens":["v"],
                "operator_tokens":[],
                "prefix_expression": ["v"],
            }
        },
        {
            "token_name": "delta_v",
            "token_type": "input_variable",
            "is_basic": True,
            "description": "speed difference between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": -1
            },
            "recipes":{
                "variable_tokens":["delta_v"],
                "operator_tokens":[],
                "prefix_expression": ["delta_v"],
            }
        }
    ],
    "fixed_const_tokens": [
        {
            "token_name": "1",
            "token_type": "fixed_const",
            "is_basic": True,
            "description": "constant 1",
            "physical_units": {
                "m": 0,
                "s": 0
            },
            "recipes":{
                "variable_tokens":["1"],
                "operator_tokens":[],
                "prefix_expression": ["1"],
            }
        }
    ],
    "free_const_tokens": [
        {
            "token_name": "v0",
            "token_type": "free_const",
            "is_basic": True,
            "description": "desired speed of ego",
            "physical_units": {
                "m": 1,
                "s": -1
            },
            "recipes":{
                "variable_tokens":["v0"],
                "operator_tokens":[],
                "prefix_expression": ["v0"],
            }
        },
        {
            "token_name": "s0",
            "token_type": "free_const",
            "is_basic": True,
            "description": "desired following distance between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": 0
            },
            "recipes":{
                "variable_tokens":["s0"],
                "operator_tokens":[],
                "prefix_expression": ["s0"],
            }
        },
        {
            "token_name": "alpha",
            "token_type": "free_const",
            "is_basic": True,
            "description": "max acceleration of ego",
            "physical_units": {
                "m": 1,
                "s": -2
            },
            "recipes":{
                "variable_tokens":["alpha"],
                "operator_tokens":[],
                "prefix_expression": ["alpha"],
            }
        },
        {
            "token_name": "k",
            "token_type": "free_const",
            "is_basic": True,
            "description": "sensitivity of driver's acceleration changes",
            "physical_units": {
                "m": 0,
                "s": 0
            },
            "recipes":{
                "variable_tokens":["k"],
                "operator_tokens":[],
                "prefix_expression": ["k"],
            }
        }
    ]
}
```