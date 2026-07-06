from abc import ABC
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from copy import copy
from time import time

from .algebra import indices_to_gf2_matrix
from .probability import ProbabilityDistribution, FactorizedDistribution, LinCombApproximation
from .metrics import MMD
    
class Trainer(ABC):
    pass

class MaxCoefficientHeuristic(Trainer):

    def __init__(self):
        self.reset()

    def reset(self):
        self.independent_parities = []
        self.coefficients = []
        self.sorted_ks = []
        self.sorted_spectrum = []

    def is_independent(self, candidate, n) -> bool:
        set = self.independent_parities + [candidate]
        matrix = indices_to_gf2_matrix(set, n)
        return np.linalg.matrix_rank(matrix) == len(set)

    def sort_contributions(self, ks, spectrum):
        
        permutation = np.argsort(np.abs(spectrum))[::-1]
        return list(ks[permutation]), list(spectrum[permutation])

    def fit(self, target:ProbabilityDistribution, hw_min:int, hw_max:int) -> FactorizedDistribution:
        
        self.sorted_ks, self.sorted_spectrum = self.sort_contributions(
            target.ks(hw_min, hw_max),
            target.walsh_hadamard_spectrum(hw_min, hw_max),
        )
    
        count = 0
        for k, c in zip(self.sorted_ks, self.sorted_spectrum):
            if self.is_independent(k, target.n):
                self.independent_parities.append(k)
                self.coefficients.append(np.clip(c, -1, 1))
                count += 1

            if count == target.n:
                return FactorizedDistribution(self.independent_parities, [0.5*(1+c) for c in self.coefficients])
            
        raise ValueError("Could not find enough indipendent parities, consider broadening the spectrum.")
    
class OptimizedMaxCoeffHeuristic(MaxCoefficientHeuristic):

    def fit(self, target:ProbabilityDistribution, mmd: MMD, maxiter:int = 10) -> FactorizedDistribution:

        # Greedy-selection of the generators
        p = super().fit(target, mmd.hw_min, mmd.hw_max)
        
        def loss(marginals):
            p.probabilities = marginals
            return mmd(target, p)

        minimize(
            loss,
            p.probabilities,
            method='L-BFGS-B', 
            bounds=[(0, 1)] * p.n, # Prevent extreme values
            options={'maxiter': maxiter}
        )

        return p
    
class OptimizedOnlyCoeffs(Trainer):

    def reset(self):
        return
    
    def fit(self, target:ProbabilityDistribution, mmd: MMD, maxiter:int = 10) -> FactorizedDistribution:

        probs = list(np.clip(0.5*(1+target.walsh_hadamard_spectrum(1,1)), 0, 1))
        p = FactorizedDistribution([1<<i for i in range(target.n)], probs)
        
        def loss(marginals):
            p.probabilities = marginals
            return mmd(target, p)

        minimize(
            loss,
            p.probabilities,
            method='L-BFGS-B', 
            bounds=[(0, 1)] * p.n, # Prevent extreme values
            options={'maxiter': maxiter}
        )

        return p
    
class MaxLinearCombHeuristic(Trainer):

    def __init__(self, n_probs:int):
        self.n_probs = n_probs 
        return super().__init__()

    def fit(self, target:ProbabilityDistribution, mmd:MMD) -> LinCombApproximation:

        max_factorized = MaxCoefficientHeuristic()
        factorized = max_factorized.fit(target, mmd.hw_min, mmd.hw_max)
        result = LinCombApproximation([factorized], [1.0])
        
        for i in range(2,self.n_probs+1):

            current = LinCombApproximation([target, result], [1.0, -1.0])

            max_factorized.reset()
            factorized = max_factorized.fit(current,  mmd.hw_min, mmd.hw_max)
            result.append(factorized, 1/i)
            
        return result

class OptimizedMaxLinearCombHeuristic(MaxLinearCombHeuristic):

    def fit(self, target:ProbabilityDistribution, mmd: MMD, maxiter:int = 10, verobse:bool = False) -> LinCombApproximation:

        # Greedy-selection of the generators
        approximation = super().fit(target, mmd)
        
        initial_parameters = []
        for dist in approximation.probabilities:
            for p in dist.probabilities:
                initial_parameters.append(p)
        
        def loss(params):
            params = np.clip(params, 0, 1)
            for i, dist in enumerate(approximation.probabilities):
                dist.probabilities = params[dist.n*i:dist.n*(i+1)]
            return mmd(target, approximation)

        def callback(res:OptimizeResult):
            print(f"{res}")

        minimize(
            loss,
            initial_parameters,
            method='L-BFGS-B', 
            bounds=[(0, 1)] * approximation.n * self.n_probs, # Prevent extreme values
            options={'maxiter': maxiter},
            callback=callback if verobse else None,
        )

        return approximation


class SweepingLinearCombHeuristic(Trainer):

    def __init__(self, n_probs:int):
        self.n_probs = n_probs 
        return super().__init__()

    def sweep(self, target:ProbabilityDistribution, current:LinCombApproximation, mmd:MMD, maxiter:int):
        
        ps = copy(current.probabilities)
        ws = copy(current.weights)
        
        for i in range(self.n_probs):

            p = ps.pop(i)
            w = ws[i]
            reminder = LinCombApproximation([target] + ps, [1/w] + (self.n_probs-1)*[-1])
            self.single_prob_optimizer.reset()
            new_p = self.single_prob_optimizer.fit(reminder, mmd, maxiter)
            ps.insert(i, new_p)
        
        return LinCombApproximation(ps, ws)

    def fit(
        self,
        target:ProbabilityDistribution, 
        mmd:MMD,
        sweeps:int = 10, 
        it_per_sweep:int = 10,
        verbose:bool = False,
        fit_generators:bool = True,
        save_history:bool = False,
    ) -> LinCombApproximation:

        self.single_prob_optimizer = OptimizedMaxCoeffHeuristic() if fit_generators else OptimizedOnlyCoeffs()

        current = LinCombApproximation(
            self.n_probs*[
                FactorizedDistribution([1<<i for i in range(target.n)], target.n*[0.5])
            ],
            self.n_probs*[1/self.n_probs],
        )
        
        self.history = []
        for i in range(sweeps):
            if verbose: 
                t0_sweep = time()
                print(f"Sweep {i+1}..", flush=True, end="")
            if save_history: 
                self.history.append(mmd(target, current))
            
            current = self.sweep(target, current, mmd, it_per_sweep)
            
            if verbose:
                print(f"Done [{(time()-t0_sweep):.2f}s] - MMD={self.history[-1]:.3e}", flush=True, )

        if save_history: 
            self.history.append(mmd(target, current))

        return current
