from qfso.models.approximate import (
    FactorizedDistribution,
    LinCombApproximation, 
    MaxCoefficientHeuristic, 
    OptimizedMaxCoeffHeuristic, 
    MaxLinearCombHeuristic,
    OptimizedMaxLinearCombHeuristic,
    SweepingLinearCombHeuristic,
    DiscretizedGaussian,
    MMD
)

from qfso.distributions import plot_distributions
from time import time


def print_factorized(p, name):
    print(f"{name}.probabilities = {p.probabilities}")
    print(f"{name}.independent_parities = {p.independent_parities}")
    print(20*"=")

def plot_training_curve(history):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.semilogy(range(len(history)), history)
    plt.xlabel("sweeps")
    plt.ylabel("MMD^2")
    plt.title("Training curve")
    plt.show()

    

if __name__ == "__main__":

    n = 6
    hw_min = 1
    hw_max = 2
    n_probs = 4
    maxiter = 4

    g1 = DiscretizedGaussian(n=n, loc=-2.3, scale=0.13)
    g2 = DiscretizedGaussian(n=n, loc=0.2, scale=0.6)
    g3 = DiscretizedGaussian(n=n, loc=2.5, scale=0.1)

    target = LinCombApproximation([g1,g2,g3], [0.4,0.1,0.5])
    mmd = MMD(sigma=0.25*n, hw_min=hw_min, hw_max=hw_max)

    target.walsh_hadamard_spectrum(mmd.hw_min, mmd.hw_max)
    #trainer = MaxCoefficientHeuristic()
    #result = trainer.fit(target, hw_min, hw_max)
    
    #print(20*"=")
    #print(f"MMD = {mmd(target, result)}")
    #print_factorized(result, "result")

    #trainer = OptimizedMaxCoeffHeuristic()
    #result = trainer.fit(target, mmd)
    
    #print(f"MMD = {mmd(target, result)}")
    #print_factorized(result, "result_optimized")

    #trainer = MaxLinearCombHeuristic(n_probs)
    #result = trainer.fit(target, mmd)

    #print(mmd(target, result))
    #for i,p in enumerate(result.probabilities):
    #    print_factorized(p, f"p[{i}]")

    print("Start optimization...", flush=True)
    t0 = time()
    #trainer = OptimizedMaxLinearCombHeuristic(n_probs)
    trainer = SweepingLinearCombHeuristic(n_probs)
    result = trainer.fit(target, mmd, sweeps = 10, it_per_sweep=maxiter, verbose=True, fit_generators=True, save_history=True)
    t1 = time()
    
    training_error = mmd(target, result)
    print(f"Done [{t1-t0:.2f}s] - Error = {training_error:.3e}")
   

    plot_distributions([target.vector(), result.vector()], ["target", f"result (mmd = {training_error:.4e})"], "test")
    plot_training_curve(trainer.history)
    
    #for i,p in enumerate(result.probabilities):
    #    print_factorized(p, f"p[{i}]")
