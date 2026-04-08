from importlib import import_module

_EXPORTS = {
    "median_heuristic": ("qfso.sigma", "median_heuristic"),
    "sigma_spectrum": ("qfso.sigma", "sigma_spectrum"),
    "sigma_heuristic": ("qfso.euristics", "sigma_heuristic"),
    "expvals_contraction": ("qfso.expectation", "expvals_contraction"),
    "expvals_sampling": ("qfso.expectation", "expvals_sampling"),
    "expvals_mc": ("qfso.expectation", "expvals_mc"),
    "run_metropolis": ("qfso.distributions.ising_generator", "run_metropolis"),
    "generate_distribution_with_target_entropy": (
        "qfso.distributions.boltzman_entropy_generator",
        "generate_distribution_with_target_entropy",
    ),
    "sample_dataset_from_distribution": (
        "qfso.distributions.boltzman_entropy_generator",
        "sample_dataset_from_distribution",
    ),
    "mmd_mc": ("qfso.mmd", "mmd_mc"),
    "local_gates": ("qfso.models", "local_gates"),
    "RStringZ": ("qfso.models", "RStringZ"),
    "IQPTensorNetwork": ("qfso.models", "IQPTensorNetwork"),
    "convert_to_jnp_ndarray": ("qfso.utils", "convert_to_jnp_ndarray"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'qfso' has no attribute '{name}'")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, symbol_name)


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))

__all__ = [
    "median_heuristic",
    "sigma_spectrum",
    "expvals_contraction",
    "expvals_sampling",
    "expvals_mc",
    "run_metropolis",
    "mmd_mc",
    "sigma_heuristic",
    "local_gates",
    "RStringZ",
    "IQPTensorNetwork",
    "convert_to_jnp_ndarray",
    "generate_distribution_with_target_entropy",
    "sample_dataset_from_distribution",
]
