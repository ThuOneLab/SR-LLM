"""
General Symbolic Regression with Physical Units Constraints and RAG

This module provides a high-level interface for performing symbolic regression
given data distributions, variable names, physical units, and semantic descriptions.
It leverages the SR-LLM framework (RL + LLM + RAG) to discover interpretable formulas.

Usage Example:
--------------
import numpy as np
from codes.applications.general_symbolic_regression import general_symbolic_regression

# Generate synthetic data: y = x1^2 + 2*x2 + 1
np.random.seed(42)
X = np.random.rand(1000, 2) * 10
y = X[:, 0]**2 + 2*X[:, 1] + 1

# Run symbolic regression with physical meaning
best_expr, best_func = general_symbolic_regression(
    X, y,
    variable_names=["x1", "x2"],
    variable_units=[[0, 0], [0, 0]],  # dimensionless
    variable_descriptions=["first input variable", "second input variable"],
    target_name="y",
    target_unit=[0, 0],
    target_description="output variable",
    seed=100,
    n_epochs=30,
    n_evolutions=8,
)
print("Best expression:", best_expr)
"""

import copy
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from codes.trafficSR.E_train_codes.RL_SAC_LLM.RL_SAC_LLM import RL_LLM_train
from codes.trafficSR.E_train_codes.train_utils.RLtrainUtils import setup_seed

# ---------------------------------------------------------------------------
# Default configurations inspired by RANDOM benchmark settings in SR-LLM
# ---------------------------------------------------------------------------

DEFAULT_OPERATOR_TOKENS = [
    "add", "mul", "sub", "div",
    "n2", "sqrt",
    "sin", "cos",
    "exp", "log",
    "pow",
]

DEFAULT_OPERATOR_DESCRIPTION = {
    "add": "addition",
    "mul": "multiplication",
    "sub": "subtraction",
    "div": "division",
    "n2": "square",
    "n3": "cube",
    "n4": "fourth power",
    "sqrt": "square root",
    "sin": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "exp": "exponential",
    "log": "logarithm",
    "pow": "power",
    "abs": "absolute value",
    "neg": "negative",
    "inv": "inverse",
}

DEFAULT_FREE_CONST_BOUNDS = [-10.0, 10.0]


def _pad_units(unit_vec: Union[List[float], np.ndarray], size: int = 7) -> List[float]:
    """Pad a short unit vector to `size` dimensions with zeros.
    SI base units order: [m, s, kg, K, A, cd, mol].
    """
    unit_vec = list(unit_vec)
    if len(unit_vec) < size:
        unit_vec = unit_vec + [0.0] * (size - len(unit_vec))
    return unit_vec[:size]


def build_train_args(
    variable_names: List[str],
    variable_units: List[List[float]],
    variable_descriptions: List[str],
    target_name: str,
    target_unit: List[float],
    target_description: str,
    operator_tokens: List[str],
    free_const_tokens: List[str],
    free_const_units: List[List[float]],
    free_const_descriptions: List[str],
    free_const_bounds: List[List[float]],
    fixed_const_tokens: List[str],
    fixed_const_values: List[float],
    fixed_const_units: List[List[float]],
    fixed_const_descriptions: List[str],
    n_epochs: int,
    n_evolutions: int,
    use_rag: bool,
    device: str,
    address: str,
    port: str,
) -> Dict:
    """
    Build a complete `train_args` dictionary compatible with `RL_LLM_train`.
    """
    # Pad all units to 7 dimensions
    variable_units_padded = {name: _pad_units(u) for name, u in zip(variable_names, variable_units)}
    free_const_units_padded = {name: _pad_units(u) for name, u in zip(free_const_tokens, free_const_units)}
    fixed_const_units_padded = {name: _pad_units(u) for name, u in zip(fixed_const_tokens, fixed_const_units)}
    target_unit_padded = _pad_units(target_unit)

    # Build variable description dict
    variable_description_dict = {}
    for name, desc in zip(variable_names, variable_descriptions):
        variable_description_dict[name] = desc

    # Build free const description dict
    free_const_description_dict = {}
    for name, desc in zip(free_const_tokens, free_const_descriptions):
        free_const_description_dict[name] = desc

    # Build free const initial values (default 1.0)
    free_const_initial_dict = {name: 1.0 for name in free_const_tokens}

    # Build free const bounds dict
    free_const_bounds_dict = {}
    for name, bounds in zip(free_const_tokens, free_const_bounds):
        free_const_bounds_dict[name] = list(bounds)

    # Build fixed const value / description dicts
    fixed_const_values_dict = {name: float(val) for name, val in zip(fixed_const_tokens, fixed_const_values)}
    fixed_const_description_dict = {name: desc for name, desc in zip(fixed_const_tokens, fixed_const_descriptions)}

    # Operator description subset
    operator_description = {op: DEFAULT_OPERATOR_DESCRIPTION.get(op, op) for op in operator_tokens}

    # ------------------------------------------------------------------
    # token_args
    # ------------------------------------------------------------------
    token_args = {
        "operator_tokens": operator_tokens,
        "operator_description": operator_description,
        "variable_tokens": variable_names,
        "variable_description": variable_description_dict,
        "variable_units": variable_units_padded,
        "free_const_tokens": free_const_tokens,
        "free_const_description": free_const_description_dict,
        "free_const_initial_values_dict": free_const_initial_dict,
        "free_const_units": free_const_units_padded,
        "free_const_bounds": free_const_bounds_dict,
        "semi_free_const_tokens": [],
        "semi_free_const_description": {},
        "semi_free_const_initial_values_dict": {},
        "semi_free_const_units": {},
        "semi_free_const_bounds": {},
        "fixed_const_tokens": fixed_const_tokens,
        "fixed_const_values_dict": fixed_const_values_dict,
        "fixed_const_units": fixed_const_units_padded,
        "fixed_const_description": fixed_const_description_dict,
        "combination_tokens": [],
        "combination_description": {},
        "combination_units": {},
        "combination_prefix_expression": {},
    }

    # ------------------------------------------------------------------
    # library_args
    # ------------------------------------------------------------------
    library_args = {
        "tokens_args": token_args,
        "min_occupancy_times": {"targets_name": [], "min_times": []},
        "superparent_units": target_unit_padded,
        "superparent_names": [target_name],
        "superparent_prog": [],  # empty for general SR; framework will auto-fill
    }

    # ------------------------------------------------------------------
    # env_args
    # ------------------------------------------------------------------
    # Heuristic max_time_step based on number of variables
    n_vars = len(variable_names)
    base_length = 20 + n_vars * 5
    max_time_step = min(max(base_length, 25), 50)

    env_args = {
        "device": device,
        "dtype": torch.float32,
        "batch_size": 1000,
        "max_time_step": max_time_step,
        "gp_gamma_decay": 1.0,
        "entropy_gamma_decay": 0.7,
        "entropy_weight": 0.005,
        "parallel_mode": False,
        "n_cpus": 6,
        "risk_factor": 0.05,
        "similarity_args": {
            "compute_similarity": True,
            "similarity_compute_limit": 0.0,
            "similarity_reward": None,
        },
        "reward_weight": {
            "factor_complexity": 0.0,
            "factor_similarity": 0.0,
            "factor_rmse": 1.0,
        },
    }

    # ------------------------------------------------------------------
    # prior_args
    # ------------------------------------------------------------------
    prior_type = [
        "HardLength",
        "SoftLength",
        "SoftMaxLength",
        "NoneSingleArityInverse",
        "NoneDoubleArityInverse",
        "PhysicalUnits",
        "NoneNested",
    ]

    # Auto-detect if Power prior is needed
    if "pow" in operator_tokens:
        prior_type.append("Power")

    prior_args = {
        "prior_type": prior_type,
        "prior_config": {
            "LengthPrior": {"min_length": 4, "max_length": None},
            "SoftLengthPrior": {
                "length_loc": int(max_time_step / 2),
                "max_length_loc": int(max_time_step / 2),
                "scale": int(max_time_step / 2) / 1.3,
                "eps": 1e-2,
            },
            "PhysicalUnitsPrior": {"prob_eps": np.finfo(np.float32).eps},
            "FirstNotFourOperatorsPrior": {"NotFour": False},
        },
    }

    # ------------------------------------------------------------------
    # agent_args
    # ------------------------------------------------------------------
    agent_args = {
        "fewshot_num": 2 if use_rag else 0,
        "best_symbol_num": 2,
        "delete_combination_num": 1,
        "best_expression_num": 5,
        "reflection_num": 1,
        "extend_num": 1,
        "max_try_num": 3,
    }

    # ------------------------------------------------------------------
    # bool_args
    # ------------------------------------------------------------------
    bool_args = {
        "bool_use_true_trajectory": False,
        "n_epochs": n_epochs,
        "epochs_save_expression_number": 15,
        "bool_use_UCB_for_sampling": True,
        "bool_plot_intermediate_process": True,
        "plot_intermediate_process_limit": 0.95,
        "n_workers": 1,
        "bool_DRL_experience_replay": False,
        "experience_replay_rate": 0.02,
        "retain_data_rounds": 3,
        "exp_decay_epsilon_greedy_setting": {
            "decay_ratio": 0.99,
            "init_epsilon": 1.0,
            "min_epsilon": 0.5,
        },
        "bool_use_rough_calibration": False,
        "calibration_conversion_limit": 0.6,
        "bool_select_k_combinations": True,
        "n_selection_of_combinations": 8,
        "bool_compute_numerical_score": True,
        "bool_compute_semantic_score": True,
        "bool_use_combinations": True,
        "bool_use_evolutions": n_evolutions > 1,
        "n_evolutions": max(n_evolutions, 1),
        "bool_extract_valuable": False,
        "bool_agent_single_step_increase": True,
        "new_tokens_number": 10,
        "bool_reflection": use_rag,
        "bool_explain_final_expressions": False,
        "bool_find_new_formula": True,  # Allow LLM to freely generate new symbol combinations
        "bool_is_feynman": False,
        "bool_is_random": True,  # Use Prompt_Generator_Random for general symbolic regression
    }

    # ------------------------------------------------------------------
    # model / optimizer / early_stop / sac args
    # ------------------------------------------------------------------
    model_type = "combined_lstm"
    model_args = {
        "hidden_size": (128, 128),
        "embedding_dim": (256, 256),
        "n_layers": (1, 1),
        "overall_and_partial": (False, True),
    }

    optimizer_args = {
        "type": "Adam",
        "lr": 0.0025,
    }

    early_stop_args = {
        "stop_reward": 0.99974,
        "stop_after_n_epochs": 1,
    }

    sac_value_args = {
        "hidden_size": (256, 256),
        "n_layers": (1, 1),
        "overall_and_partial": (False, True),
    }

    sac_args = {
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "alpha_lr": 3e-4,
        "tau": 0.05,
        "gamma": 0.99,
        "n_warmup_batches": 1,
        "sample_batch_size": env_args["batch_size"] * 5,
        "replay_buffer_capacity": int(
            env_args["risk_factor"]
            * env_args["batch_size"]
            * env_args["max_time_step"]
            / 3
            * 5
        ),
    }

    train_args = {
        "model_type": model_type,
        "model_args": model_args,
        "optimizer_args": optimizer_args,
        "early_stop_args": early_stop_args,
        "env_args": env_args,
        "prior_args": prior_args,
        "token_args": token_args,
        "library_args": library_args,
        "sac_value_args": sac_value_args,
        "sac_args": sac_args,
        "agent_args": agent_args,
        "bool_args": bool_args,
        "address": address,
        "port": port,
    }

    return train_args


def general_symbolic_regression(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    variable_names: List[str],
    variable_units: List[List[float]],
    variable_descriptions: List[str],
    target_name: str = "y",
    target_unit: List[float] = None,
    target_description: str = "The target variable to be predicted",
    operator_tokens: Optional[List[str]] = None,
    free_const_tokens: Optional[List[str]] = None,
    free_const_units: Optional[List[List[float]]] = None,
    free_const_descriptions: Optional[List[str]] = None,
    free_const_bounds: Optional[List[List[float]]] = None,
    fixed_const_tokens: Optional[List[str]] = None,
    fixed_const_values: Optional[List[float]] = None,
    fixed_const_units: Optional[List[List[float]]] = None,
    fixed_const_descriptions: Optional[List[str]] = None,
    seed: int = 100,
    n_epochs: int = 50,
    n_evolutions: int = 10,
    use_rag: bool = True,
    memory_path: str = "codes/ragLibrary/memory_general",
    device: str = "cpu",
    address: str = "172.22.0.1",
    port: str = "7890",
) -> Tuple[str, callable]:
    """
    Perform general symbolic regression using the SR-LLM framework.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (n_samples, n_features)
        Input data matrix.
    y : np.ndarray or torch.Tensor, shape (n_samples,)
        Target values.
    variable_names : List[str]
        Names of input variables, e.g., ["x1", "x2", "temperature"].
    variable_units : List[List[float]]
        Physical unit vectors for each variable (SI base: [m, s, kg, K, A, cd, mol]).
        Short vectors will be zero-padded to 7 dimensions automatically.
    variable_descriptions : List[str]
        Human-readable descriptions of each variable.
    target_name : str, optional
        Name of the target variable. Default is "y".
    target_unit : List[float], optional
        Physical unit vector of the target. Default is [0, 0] (dimensionless).
    target_description : str, optional
        Description of the target variable.
    operator_tokens : List[str], optional
        Allowed operators. Default includes basic arithmetic + transcendental functions.
    free_const_tokens : List[str], optional
        Names of free constants to be optimized. Defaults to ["c_1", "c_2", "c_3"].
    free_const_units : List[List[float]], optional
        Physical units of free constants. Defaults to dimensionless.
    free_const_descriptions : List[str], optional
        Descriptions of free constants.
    free_const_bounds : List[List[float]], optional
        Box constraints for each free constant. Defaults to [-10, 10].
    fixed_const_tokens : List[str], optional
        Names of fixed constants (e.g., ["1", "pi"]).
    fixed_const_values : List[float], optional
        Numerical values of fixed constants.
    fixed_const_units : List[List[float]], optional
        Physical units of fixed constants.
    fixed_const_descriptions : List[str], optional
        Descriptions of fixed constants.
    seed : int, optional
        Random seed for reproducibility. Default is 100.
    n_epochs : int, optional
        Number of DRL epochs per evolution. Default is 50.
    n_evolutions : int, optional
        Number of LLM-driven evolutions (set to 1 to disable LLM expansion). Default is 10.
    use_rag : bool, optional
        Whether to enable RAG (Retrieval-Augmented Generation) for LLM symbol expansion.
        If True, a `memory_path` must point to a valid RAG library (see docs/README_RAG.md).
    memory_path : str, optional
        Path to the Chroma-based RAG knowledge pool. Default is "codes/ragLibrary/memory_general".
    device : str, optional
        PyTorch device, "cpu" or "cuda". Default is "cpu".
    address : str, optional
        Proxy address for OpenAI API. Default is "172.22.0.1".
    port : str, optional
        Proxy port for OpenAI API. Default is "7890".

    Returns
    -------
    best_expression : str
        The best discovered symbolic expression (simplified infix form).
    best_function : callable
        A torch-compatible function that evaluates the discovered expression on new data.
        Input should be a torch.Tensor of shape (n_samples, n_features).
    """
    setup_seed(seed)

    # Convert inputs to torch tensors
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.float() if X.dtype != torch.float32 else X

    if isinstance(y, np.ndarray):
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        y_tensor = y.float() if y.dtype != torch.float32 else y

    # Ensure y is 1-D
    if y_tensor.dim() > 1:
        y_tensor = y_tensor.squeeze()

    n_vars = len(variable_names)
    assert X_tensor.shape[1] == n_vars, (
        f"Number of columns in X ({X_tensor.shape[1]}) must match "
        f"number of variable_names ({n_vars})"
    )
    assert len(variable_units) == n_vars, (
        f"Length of variable_units ({len(variable_units)}) must match "
        f"number of variable_names ({n_vars})"
    )
    assert len(variable_descriptions) == n_vars, (
        f"Length of variable_descriptions ({len(variable_descriptions)}) must match "
        f"number of variable_names ({n_vars})"
    )

    # Default target unit
    if target_unit is None:
        target_unit = [0.0, 0.0]

    # Default operators
    if operator_tokens is None:
        operator_tokens = DEFAULT_OPERATOR_TOKENS.copy()

    # Default free constants
    if free_const_tokens is None:
        free_const_tokens = [f"c_{i + 1}" for i in range(3)]
    n_free = len(free_const_tokens)

    if free_const_units is None:
        free_const_units = [[0.0, 0.0] for _ in range(n_free)]
    if free_const_descriptions is None:
        free_const_descriptions = [f"free constant {i + 1}" for i in range(n_free)]
    if free_const_bounds is None:
        free_const_bounds = [DEFAULT_FREE_CONST_BOUNDS.copy() for _ in range(n_free)]

    # Default fixed constants
    if fixed_const_tokens is None:
        fixed_const_tokens = ["1"]
    n_fixed = len(fixed_const_tokens)

    if fixed_const_values is None:
        fixed_const_values = [1.0 for _ in range(n_fixed)]
    if fixed_const_units is None:
        fixed_const_units = [[0.0, 0.0] for _ in range(n_fixed)]
    if fixed_const_descriptions is None:
        fixed_const_descriptions = [f"fixed constant {i + 1}" for i in range(n_fixed)]

    # Build configuration
    train_args = build_train_args(
        variable_names=variable_names,
        variable_units=variable_units,
        variable_descriptions=variable_descriptions,
        target_name=target_name,
        target_unit=target_unit,
        target_description=target_description,
        operator_tokens=operator_tokens,
        free_const_tokens=free_const_tokens,
        free_const_units=free_const_units,
        free_const_descriptions=free_const_descriptions,
        free_const_bounds=free_const_bounds,
        fixed_const_tokens=fixed_const_tokens,
        fixed_const_values=fixed_const_values,
        fixed_const_units=fixed_const_units,
        fixed_const_descriptions=fixed_const_descriptions,
        n_epochs=n_epochs,
        n_evolutions=n_evolutions,
        use_rag=use_rag,
        device=device,
        address=address,
        port=port,
    )

    # ngsim_args for general regression (single data source)
    ngsim_args = {
        "n_data_sources": 1,
        "stop_positions": [len(X_tensor) - 1],
    }

    print("=" * 60)
    print("General Symbolic Regression Started")
    print("=" * 60)
    print(f"Variables      : {variable_names}")
    print(f"Target         : {target_name}")
    print(f"Data shape     : X={X_tensor.shape}, y={y_tensor.shape}")
    print(f"Epochs         : {n_epochs}")
    print(f"Evolutions     : {n_evolutions}")
    print(f"RAG enabled    : {use_rag}")
    print(f"Memory path    : {memory_path}")
    print("=" * 60)

    # Run SR-LLM training
    R_last_epoch, best_expression, best_function = RL_LLM_train(
        X=X_tensor,
        y=y_tensor,
        train_args=train_args,
        ngsim_args=ngsim_args,
        data_source=None,
        seed=seed,
        memory_path=memory_path,
    )

    if best_expression is None:
        print("Warning: No valid expression was discovered.")
        return None, None

    print("=" * 60)
    print("Best discovered expression:", best_expression)
    print("=" * 60)

    return str(best_expression), best_function


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Simple demonstration: recover y = x1^2 + 2*x2 + 1
    # ------------------------------------------------------------------
    setup_seed(42)
    n_samples = 1000
    X_np = np.random.rand(n_samples, 2) * 10
    y_np = X_np[:, 0] ** 2 + 2 * X_np[:, 1] + 1

    expr, func = general_symbolic_regression(
        X_np,
        y_np,
        variable_names=["x1", "x2"],
        variable_units=[[0, 0], [0, 0]],
        variable_descriptions=["first dimensionless input", "second dimensionless input"],
        target_name="y",
        target_unit=[0, 0],
        target_description="output variable",
        seed=100,
        n_epochs=10,      # small for quick demo
        n_evolutions=2,   # small for quick demo
        use_rag=False,    # disable RAG for demo (no LLM API key needed)
        device="cpu",
    )
    print("Demo result expression:", expr)
