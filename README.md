# qfso

Quantum Fourier Space Operations.

`qfso` provides tools to work with qft circuits and probability distributions and dataset generation utilities used in current experiments.

## Package Structure

The package is organized in two main namespaces:

- `qfso.distributions`
	- `generate`: dataset/probability generators
	- `transform.wh`: Walsh-Hadamard transform and decomposition utilities
- `qfso.models`
	- `iqp`: IQP circuit model and operations on IQP circuits

The utility module `qfso.utils` remains available for shared helper functions.

## Current Functionalities

### 1. IQP modeling and operations (`qfso.models.iqp`)

- IQP circuit construction:
	- `IQPTensorNetwork`
	- `local_gates`
	- `RStringZ`
- Expectation estimation:
	- `expvals_contraction`
	- `expvals_sampling`
	- `expvals_mc`
- MMD loss and optimization:
	- `mmd_mc`
	- `setup_training`
- Sigma heuristics:
	- `median_heuristic`
	- `sigma_spectrum`
	- `sigma_heuristic`

### 2. Distribution generation (`qfso.distributions.generate`)

- Ising Metropolis sampler:
	- `run_metropolis`
- Entropy-controlled synthetic distribution tools:
	- `generate_distribution_with_target_entropy`
	- `sample_dataset_from_distribution`

### 3. Walsh-Hadamard transforms (`qfso.distributions.transform.wh`)

- Basis functions and decomposition classes:
	- `BasisFunction`
	- `WalshHadamardBasisFunction`
	- `FourierDecomposition`
	- `WalshHadamardDecomposition`
- Utility functions:
	- `WH_fixed_order_ids`
	- `discretized_normal_probability`
	- `exact_WH_coefficient`
	- `convergence_scaling`

## Installation

Minimal install:

```bash
pip install -e .
```

With IQP/training dependencies:

```bash
pip install -e .[iqp]
```

With notebook dependencies:

```bash
pip install -e .[notebooks]
```

Install everything:

```bash
pip install -e .[all]
```

## Quick Usage

```python
from qfso.models.iqp import IQPTensorNetwork, local_gates, setup_training, sigma_spectrum
from qfso.distributions.generate import run_metropolis
from qfso.distributions.transform.wh import WalshHadamardDecomposition
```

Top-level convenience imports are also available for the main public symbols:

```python
from qfso import IQPTensorNetwork, mmd_mc, sigma_heuristic
```

## Command Line

Generate Ising samples from CLI:

```bash
qfso-ising 6 --all_steps 100000 --temp 2.4 --h 0.08
```

This runs the Ising Metropolis generator and saves a `.npy` dataset.

## Status

The project is under active development. Additional models, operations, and transforms will be added over time.
