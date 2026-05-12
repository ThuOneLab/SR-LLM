# Official implementation of SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation

## 📢 News
- ✅ **2026-05-12**: Added **General Symbolic Regression** functionality — now supports arbitrary formulas with physical unit constraints and RAG-enhanced LLM assistance.
- ✅ **2025-11-14**: Released result files of SR-LLM on different benchmarks, and support for testing SR-LLM in discovering new car-following models.
- 🚀 Code for benchmark evaluations is currently being organized.

More code is coming soon!

## Overview

This repository hosts the **official release** of the implementation code for the paper:  
**SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation**.

## Installation

First, clone the repository:

```bash
git clone https://github.com/your-org/SR-LLM.git
cd SR-LLM
```

This repository has been tested on **Linux (Ubuntu 24.04)**. Follow the instructions below to set up the environment.

### Linux (Ubuntu 24.04)

```bash
conda create -n sr-llm python=3.10.18
conda env update --file environment.yml
conda activate sr-llm
pip install torch torchvision
```

## Usage

### Fundamental-Benchmark (🚀 Coming soon!)

The Fundamental-Benchmark experiments evaluate the basic search capability of SR-LLM **without LLM assistance** (denoted as "SR-LLM w/o" in the paper).  
To reproduce the results on this benchmark, run:

```bash
python codes/applications/evaluate_benchmark_fundamental.py
```

Turn to `results\Fundamental-Benchmark\SR-LLM-WO.csv` to see detailed results.

### Feynman-Benchmark (🚀 Coming soon!)

The Feynman-Benchmark includes two experimental settings:
- **Without LLM assistance** (SR-LLM w/o)
- **With LLM assistance** (SR-LLM)

This benchmark evaluates SR-LLM's performance on symbolic regression for meaningful equations.

To reproduce SR-LLM w/o results:

```bash
python codes/applications/evaluate_benchmark_feyn.py
```

Turn to `results\Feynman-Benchmark\SR-LLM-WO.csv` to see detailed results.

To reproduce full SR-LLM (with LLM) results:

```bash
python codes/applications/evaluate_benchmark_feyn_LLM.py
```

Turn to `results\Feynman-Benchmark\SR-LLM.csv` to see detailed results.

### Random-Benchmark (🚀 Coming soon!)

The Random-Benchmark uses synthetically generated datasets to eliminate the risk of target formulas appearing in the LLM's pretraining corpus.

To reproduce SR-LLM w/o results:

```bash
python codes/applications/evaluate_benchmark_random.py
```

Turn to `results\Random-Benchmark\SR-LLM-WO.csv` to see detailed results.

To reproduce full SR-LLM (with LLM) results:

```bash
python codes/applications/evaluate_benchmark_random_LLM.py
```

Turn to `results\Random-Benchmark\SR-LLM.csv` to see detailed results.

### Discovering New Car-Following Models from the NGSIM Dataset

We apply SR-LLM to the **NGSIM dataset**, a real-world human driving trajectory dataset, to test its ability to discover novel, interpretable, and high-performing car-following models from empirical data.

To run the experiment for discovering new models on NGSIM:

```bash
python codes/applications/SRRAG_multiprocess_new_formula.py
```

### General Symbolic Regression

Starting from **v1.1**, SR-LLM supports **general-purpose symbolic regression** on arbitrary datasets with **physical unit constraints** and **RAG-enhanced LLM assistance**.

You only need to provide:
- Input data `X` and target `y`
- Variable names, physical units, and semantic descriptions
- (Optional) Target variable name, unit, and description
- (Optional) A pre-built RAG library for your domain

#### Quick Start

```python
import numpy as np
from codes.applications.general_symbolic_regression import general_symbolic_regression

# Example: discover y = x1^2 + 2*x2 + 1
np.random.seed(42)
X = np.random.rand(1000, 2) * 10
y = X[:, 0]**2 + 2*X[:, 1] + 1

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
print("Discovered expression:", best_expr)
```

#### With Physical Unit Constraints

If your problem has physical meaning, provide SI unit vectors (`[m, s, kg, K, A, cd, mol]`). The framework will automatically enforce dimensional consistency during search:

```python
best_expr, best_func = general_symbolic_regression(
    X, y,
    variable_names=["s", "v", "delta_v"],
    variable_units=[[1, 0], [1, -1], [1, -1]],  # distance, speed, speed
    variable_descriptions=[
        "spacing between vehicles",
        "ego vehicle speed",
        "relative speed to leading vehicle",
    ],
    target_name="a",
    target_unit=[1, -2],  # acceleration
    target_description="ego vehicle acceleration",
    use_rag=True,
    memory_path="codes/ragLibrary/memory_car_following",
)
```

#### Building a RAG Library

To leverage domain knowledge, build a RAG library before running regression. See the detailed guide:

👉 **[RAG Library Construction Guide](docs/README_RAG.md)**

Key steps:
1. Initialize a `RAG_AGENT` with a `memory_path`
2. Add `Knowledge` objects describing how symbols combine in your domain
3. Save target names and pass the `memory_path` to `general_symbolic_regression`

#### Command-Line Demo

```bash
python codes/applications/general_symbolic_regression.py
```

> **Note**: The demo uses `use_rag=False` and small `n_epochs/n_evolutions` so it can run without an LLM API key. For production use, set `use_rag=True` and configure your `OPENAI_API_KEY`.

---

## Datasets Used in SR-LLM

This repository includes all formulas used in SR-LLM, including `Fundamental-Benchmark.csv` , `Feynman-Benchmark.csv` and `Random-Benchmark.csv`.
⚠️ **Note**: Some entries in `Feynman-Benchmark.csv` list an incorrect number of variables. The following equations require manual correction in the "# variables" column:
- I.18.12
- I.18.14
- III.10.19
- III.19.51

Please update the variable count based on the actual number of variables in these equations.
