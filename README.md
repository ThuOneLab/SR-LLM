# SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation

## 📢 News

### 2025.9.28
- ✅ **2025-09-28**: Released pre-review version with support for testing SR-LLM in discovering new car-following models.
- ✅ **2025-10-09**: Released result files of SR-LLM on different benchmarks.
- 🚀 Code for benchmark evaluations is currently being organized.

More code is coming soon!

## Overview

This repository hosts the **official pre-review release** of the implementation code for the paper:  
**SR-LLM: An Incremental Symbolic Regression Framework Driven by LLM-based Retrieval-Augmented Generation**.

The full, cleaned implementation will be publicly released soon after the review process.

## Installation

First, clone the repository:

```bash
wget https://anonymous.4open.science/api/repo/SR-LLM-PNAS/zip
mv zip SR-LLM.zip
unzip SR-LLM.zip -d SR-LLM
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

## Datasets Used in SR-LLM

This repository includes all formulas used in SR-LLM, including `Fundamental-Benchmark.csv` , `Feynman-Benchmark.csv` and `Random-Benchmark.csv`.
⚠️ **Note**: Some entries in `Feynman-Benchmark.csv` list an incorrect number of variables. The following equations require manual correction in the "# variables" column:
- I.18.12
- I.18.14
- III.10.19
- III.19.51

Please update the variable count based on the actual number of variables in these equations.