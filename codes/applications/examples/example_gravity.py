"""
Example 2: Discovering Newton's Law of Universal Gravitation
Target: F = G * m1 * m2 / r^2

This example demonstrates:
- Physical unit constraints (SI base units)
- Free constant optimization (G)
- Fixed constant usage (1)
- Different distributions for distinguishable variables
"""

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from codes.applications.general_symbolic_regression import general_symbolic_regression

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000

    # Use DIFFERENT distributions for m1 and m2 to make them distinguishable
    m1 = np.random.uniform(1.0, 5.0, n_samples)    # lighter object
    m2 = np.random.uniform(5.0, 20.0, n_samples)   # heavier object
    r = np.random.uniform(1.0, 5.0, n_samples)
    G_true = 1.0

    F = G_true * m1 * m2 / (r ** 2)

    X = np.column_stack([m1, m2, r])
    y = F

    print("=" * 60)
    print("Example 2: Newton's Law of Universal Gravitation")
    print("Target: F = G * m1 * m2 / r^2  (with G=1.0)")
    print("m1 range: [1.0, 5.0]  |  m2 range: [5.0, 20.0]")
    print("=" * 60)

    best_expr, best_func = general_symbolic_regression(
        X,
        y,
        variable_names=["m1", "m2", "r"],
        variable_units=[
            [0, 0, 1],   # kg
            [0, 0, 1],   # kg
            [1, 0, 0],   # m
        ],
        variable_descriptions=[
            "mass of object 1",
            "mass of object 2",
            "distance between the centers of the two objects",
        ],
        target_name="F",
        target_unit=[1, -2, 1],  # N = kg*m/s^2
        target_description="gravitational force between two objects",
        operator_tokens=["add", "mul", "sub", "div", "n2", "inv"],
        free_const_tokens=["G"],
        free_const_units=[[3, -2, -1]],  # m^3/(kg*s^2)
        free_const_descriptions=["gravitational constant"],
        free_const_bounds=[[0.1, 10.0]],
        fixed_const_tokens=["1"],
        fixed_const_values=[1.0],
        fixed_const_units=[[0, 0, 0]],
        fixed_const_descriptions=["dimensionless constant 1"],
        seed=100,
        n_epochs=30,
        n_evolutions=5,
        use_rag=False,
        device="cpu",
    )

    print("\n" + "=" * 60)
    print("TARGET:  F = G * m1 * m2 / r^2")
    print("FOUND:  ", best_expr)
    print("=" * 60)
