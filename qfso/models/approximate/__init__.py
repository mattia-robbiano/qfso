from .probability import (
    FactorizedDistribution, 
    EmpiricalDistribution, 
    LinCombApproximation,
    DiscretizedGaussian,
)
from .metrics import MMD
from .heuristics import (
    MaxCoefficientHeuristic, 
    OptimizedMaxCoeffHeuristic, 
    MaxLinearCombHeuristic,
    OptimizedMaxLinearCombHeuristic,
    SweepingLinearCombHeuristic
)