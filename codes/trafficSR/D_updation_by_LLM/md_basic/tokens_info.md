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
            "description": "following distance between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": 0
            },
        },
        {
            "token_name": "v",
            "token_type": "input_variable",
            "description": "speed of ego",
            "physical_units": {
                "m": 1,
                "s": -1
            },
        },
        {
            "token_name": "delta_v",
            "token_type": "input_variable",
            "description": "speed difference between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": -1
            },
        }
    ],
}
```