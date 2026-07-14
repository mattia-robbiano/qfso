from importlib import import_module

_EXPORTS = {
    "EmpiricalDistribution": ("qfso.models.approximate", "EmpiricalDistribution"),
    "FactorizedDistribution": ("qfso.models.approximate", "FactorizedDistribution"),
    "LinCombApproximation": ("qfso.models.approximate", "LinCombApproximation"),
    "MMD": ("qfso.models.approximate", "MMD"),
    "DiscreteGreedyFitter": ("qfso.models.approximate", "DiscreteGreedyFitter"),
    "OptimizedGreedyFitter": ("qfso.models.approximate", "OptimizedGreedyFitter"),
    "SweepingLinearCombFitter": ("qfso.models.approximate", "SweepingLinearCombFitter"),
    "IQPTensorNetwork": ("qfso.models.iqp", "IQPTensorNetwork"),
    "RStringZ": ("qfso.models.iqp", "RStringZ"),
    "local_gates": ("qfso.models.iqp", "local_gates"),
    "expvals_contraction": ("qfso.models.iqp", "expvals_contraction"),
    "expvals_sampling": ("qfso.models.iqp", "expvals_sampling"),
    "expvals_mc": ("qfso.models.iqp", "expvals_mc"),
    "mmd_mc": ("qfso.models.iqp", "mmd_mc"),
    "setup_training": ("qfso.models.iqp", "setup_training"),
    "median_heuristic": ("qfso.models.iqp", "median_heuristic"),
    "sigma_spectrum": ("qfso.models.iqp", "sigma_spectrum"),
    "sigma_heuristic": ("qfso.models.iqp", "sigma_heuristic"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'qfso.models' has no attribute '{name}'")

    module_name, symbol_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # IQP depends on optional quimb; keep approximate APIs usable without it.
        if exc.name == "quimb" and module_name.startswith("qfso.models.iqp"):
            raise ModuleNotFoundError(
                "qfso.models.iqp requires the optional dependency 'quimb'. "
                "Install it with: pip install 'qfso[iqp]'"
            ) from exc
        raise

    return getattr(module, symbol_name)


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
