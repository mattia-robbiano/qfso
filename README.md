# qfso

Quantum Fuorier Space Operations. Python package to model quantum circuits, perform Fourier transform, compute metrics and ac on them in the Fourier space.

## What is included

- circuit models: IQP (expectation-value and mmd computation routines)
- Distribution generators
- Walsh-Hadamard transform utilities

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

**Under developement!** Different models, operations and transformations will be added.
