# qfso

qfso is an installable Python package for IQP-inspired feature-space optimization workflows.

## What is included

- IQP modeling and expectation-value utilities
- MMD-based optimization helpers
- Distribution generators (including Ising sampler)
- Walsh-Hadamard decomposition utilities

## Install

Minimal core install:

```bash
pip install -e .
```

Install with IQP/training stack:

```bash
pip install -e .[iqp]
```

Install with notebook extras:

```bash
pip install -e .[notebooks]
```

Install everything:

```bash
pip install -e .[all]
```

## Quick usage

```python
from qfso.models import IQPTensorNetwork, local_gates
from qfso.sigma import sigma_spectrum
from qfso.optimizer import setup_training
```

## Command line

The package exposes one small CLI command for dataset generation:

```bash
qfso-ising 6 --all_steps 100000 --temp 2.4 --h 0.08
```

This runs the Ising Metropolis generator and saves a `.npy` dataset.

## Development note

The package source of truth is the `qfso` folder.
