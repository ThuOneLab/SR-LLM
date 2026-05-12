"""
Example 1: Discovering a polynomial formula
Target: y = x1^2 + 2*x2 + 1

This example demonstrates basic usage of general_symbolic_regression
on a dimensionless problem without physical units.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from codes.applications.general_symbolic_regression import general_symbolic_regression

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples, 2) * 10
    y = X[:, 0] ** 2 + 2 * X[:, 1] + 1

    print("=" * 60)
    print("Example 1: Polynomial Discovery")
    print("Target: y = x1^2 + 2*x2 + 1")
    print("=" * 60)

    best_expr, best_func = general_symbolic_regression(
        X,
        y,
        variable_names=["x1", "x2"],
        variable_units=[[0, 0], [0, 0]],
        variable_descriptions=["first input variable", "second input variable"],
        target_name="y",
        target_unit=[0, 0],
        target_description="output variable",
        seed=100,
        n_epochs=30,
        n_evolutions=5,
        use_rag=False,
        device="cpu",
    )

    print("\n" + "=" * 60)
    print("TARGET:  y = x1**2 + 2*x2 + 1")
    print("FOUND:  ", best_expr)
    print("=" * 60)
