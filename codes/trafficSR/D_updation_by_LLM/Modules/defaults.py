'''no use now'''
DEFAULT_CONFIG = {
    "openai_api_key": "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",
    "openai_chat_model": "gpt-4-1106-preview",  # gpt-4-1106-preview, gpt-3.5-turbo-16k-0613
    "memory_path": "memory",
    "port": "15732",
    "temperature": 0.5,
}

DEFAULT_SYMBOL_FORMAT = {
    "name": "...",
    "description": "...",
    "units": ["...", "..."],
    "prefix_expression": ["..."],
}

DEFAULT_SYMBOL_INFO = {
    "name": "v_safe",
    "description": "A speed variable used to measure the maximum speed that the ego vehicle can reach under safe conditions",
    "unit": [1, -1],
    "prefix_expression": ["div", "s", "T"],
}

DEFAULT_LIBRARY_INFO = {
    "operators": ["add", "sub", "mul", "div", "n2"],
    "variables": ["v", "v_0", "s", "v_safe"],
    "description of variables": {
        "v": "speed of ego vehicle",
        "v_0": "desired speed of ego vehicle",
        "s": "distance between ego vehicle and front vehicle",
        "T": "reaction time of ego vehicle",
        "v_safe": "A speed variable used to measure the maximum speed that the ego vehicle can reach under safe conditions"
    },
    "description of operators": {
        "add": "addition",
        "sub": "subtraction",
        "mul": "multiplication",
        "div": "division",
        "n2": "square of a number"
    },
    "physical units of variables": {
        "v": [1, -1],
        "v_0": [1, -1],
        "s": [1, 0],
        "T": [0, 1],
        "v_safe": [1, -1]
    }
}

DEFAULT_ELITE_SYMBOL_DICTINARY = {
    "key": "$s0+vT$",
    "performance": "Good",
    "prefix_expression": ["add", "s0", "mul", "v0", "T"],
    "unit": [1, 0],
    "description of operators": {
        "add": "addition",
        "mul": "multiplication",
    },
    "description of variables": {
        "s0": "minimum following distance",
        "v": "speed of ego vehicle",
        "T": "reaction time of ego vehicle",
    }
}

DEFAULT_ELITE_PROGRAM_DICTINARY = {
    "prefix_expression": ['mul', 'alpha', 'n2', 'div', 's0', 's'],
    "unit": [1, -2],
    "description_of_operators": {
        "mul": "multiplication",
        "div": "division",
        "n2": "square of a number",
    },
    "description_of_variables": {
        "alpha": "a coefficient",
        "s0": "minimum following distance",
        "s": "distance between ego vehicle and front vehicle",
    }
}

DEFAULT_SAME_MEANING_EXPRESSION_JUDGE = {
    "equivalent_expressions": True,  # True or False
    "reason": "The two expressions are the same because they are equivalent after simplification."
}
