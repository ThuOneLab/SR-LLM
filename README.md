# Official implementation of SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation

## 📢 News
- ✅ **2026-05-12**: Added **General Symbolic Regression** functionality.
- ✅ **2025-11-14**: Released result files of SR-LLM on different benchmarks, and support for testing SR-LLM in discovering new car-following models.
- 🚀 Code for benchmark evaluations is currently being organized.

More code is coming soon!

## Overview

This repository hosts the **official release** of the implementation code for the paper:  
**SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation**.

## Installation

First, clone the repository:

```bash
git clone https://github.com/ThuOneLab/SR-LLM.git
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

We provide two ready-to-run examples under `codes/applications/examples/`:

- **[example_polynomial.py](codes/applications/examples/example_polynomial.py)** — Dimensionless polynomial discovery (`y = x₁² + 2·x₂ + 1`).
- **[example_gravity.py](codes/applications/examples/example_gravity.py)** — Newton's law of universal gravitation with physical units, free constants, and fixed constants.

Run any example directly:

```bash
python codes/applications/examples/example_polynomial.py
```

#### With Physical Unit Constraints

If your problem has physical meaning, provide SI unit vectors (`[m, s, kg, K, A, cd, mol]`). The framework will automatically enforce dimensional consistency during search. See `example_gravity.py` for a complete demonstration including free constants (e.g., gravitational constant `G`) and fixed constants (e.g., `1`).

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

This repository includes all formulas used in SR-LLM, located under the `benchmarks/` folder:
- [`benchmarks/Fundamental-Benchmark.csv`](benchmarks/Fundamental-Benchmark.csv)
- [`benchmarks/Feynman-Benchmark.csv`](benchmarks/Feynman-Benchmark.csv)
- [`benchmarks/Random-Benchmark.csv`](benchmarks/Random-Benchmark.csv)

⚠️ **Note**: Some entries in `benchmarks/Feynman-Benchmark.csv` list an incorrect number of variables. The following equations require manual correction in the "# variables" column:
- I.18.12
- I.18.14
- III.10.19
- III.19.51

Please update the variable count based on the actual number of variables in these equations.
