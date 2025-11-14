#### new combinations

```python
{
    "new_combinations": {
        "v * delta_v / sqrt(alpha * b)": {
            "infix_expression": "2 * sqrt(alpha * b)",
            "description": "the maximum acceleration multiplied by the square root of the product of the acceleration and the following distance, representing the maximum acceleration that can be achieved based on comfort",
            "units": [0, 0],
            "prefix_expression": ["div", "mul", "v", "delta_v", "sqrt", "mul", "alpha", "b"]
        },
        "T * v / s": {
            "infix_expression": "T * v / s",
            "description": "time headway multiplied by the ratio of speed to following distance, representing a dynamic following distance based on comfort",
            "units": [0, 0],
            "prefix_expression": ["div", "mul", "T", "v", "s"]
        }
    }
}
```