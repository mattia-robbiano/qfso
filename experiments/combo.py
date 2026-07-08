# import pickle
# from time import time
# import numpy as np

# from qfso.models.approximate import (
#     FactorizedDistribution,
#     LinCombApproximation, 
#     DiscreteGreedyFitter, 
#     OptimizedGreedyFitter, 
#     IncrementalLinearCombBuilder,
#     GlobalOptimizedLinearComb,
#     SweepingLinearCombFitter,
#     DiscretizedGaussian,
#     FromSpectrum,
#     MMD
# )
# from qfso.distributions import plot_distributions


# def print_factorized(p, name):
#     print(f"{name}.probabilities = {p.probabilities}")
#     print(f"{name}.independent_parities = {p.independent_parities}")
#     print(20*"=")


# def plot_training_curve(history):
#     import matplotlib.pyplot as plt
#     plt.semilogy(range(len(history)), history)
#     plt.xlabel("sweeps")
#     plt.ylabel("MMD^2")
#     plt.title("Training curve")
#     plt.show()


# if __name__ == "__main__":
#     n = 20
#     hw_min = 1
#     hw_max = 2
#     n_probs = 7
#     maxiter = 50
#     sweeps = 10
#     top_n = 200 

#     g1 = DiscretizedGaussian(n=n, loc=-2.3, scale=0.13)
#     g2 = DiscretizedGaussian(n=n, loc=0.2, scale=0.6)
#     g3 = DiscretizedGaussian(n=n, loc=2.5, scale=0.1)
#     target = LinCombApproximation([g1, g2, g3], [0.4, 0.1, 0.5])

#     # spectrum = load_precomputed_fourier_coeffs('n400_target.pkl')
#     # spectrum = {}
#     # dummy_dist = FactorizedDistribution([1 << i for i in range(n)], [0.5] * n)
#     # valid_ks = dummy_dist.ks(hw_min, hw_max)
#     # for k in valid_ks:
#     # if np.random.random() > 0.7: # sparse: only 30% of ks
#     # spectrum[int(k)] = float(np.random.uniform(-0.3, 0.3))
#     # target = FromSpectrum(n=n, spectrum=spectrum)
#     # print(f"Loaded sparse spectrum: {len(spectrum)} coefficients out of {len(valid_ks)}") 
    
#     mmd = MMD(sigma=0.25*n, hw_min=hw_min, hw_max=hw_max)
    
#     print("Start optimization...", flush=True)
#     t0 = time()
    
#     trainer = SweepingLinearCombFitter(n_probs)
#     result = trainer.fit(
#         target,
#         mmd,
#         sweeps=sweeps,
#         it_per_sweep=maxiter,
#         verbose=True,
#         fit_generators=False,
#         save_history=True,
#         top_n=top_n,
#     )
#     t1 = time()
#     training_error = mmd(target, result)
#     print(f"Done [{t1-t0:.2f}s] - Error = {training_error:.3e}")
    
#     # Salvataggio del modello addestrato
#     # with open("trained_combo.pkl", "wb") as f:
#     #     pickle.dump(result, f)
#     # print("Modello salvato in 'trained_model.pkl'")

    
#     plot_training_curve(trainer.history)


import pickle
from time import time
import numpy as np

from qfso.models.approximate import (
    FactorizedDistribution,
    LinCombApproximation, 
    DiscreteGreedyFitter, 
    OptimizedGreedyFitter, 
    SweepingLinearCombFitter,
)
from qfso.models.approximate.probability.spectrum import TruncatedArraySpectrum
from qfso.models.approximate.metrics import SubsampledMMD

def plot_training_curve(history):
    import matplotlib.pyplot as plt
    plt.semilogy(range(len(history)), history)
    plt.xlabel("sweeps")
    plt.ylabel("MMD^2")
    plt.title("Training curve (Stochastic MMD)")
    plt.show()

def save_safe_model(model: LinCombApproximation, filepath: str):
    """
    Estrae solo i dati puri (pesi, parità, probabilità) disaccoppiandoli
    dall'oggetto per evitare errori di pickle con lambda o tensori JAX.
    """
    state = {
        "n": model.n,
        "weights": [float(w) for w in model.weights],
        "components": []
    }
    
    for dist in model.probabilities:
        state["components"].append({
            # Convertiamo esplicitamente in liste di interi normali
            "independent_parities": [int(k) for k in dist.independent_parities],
            # Convertiamo da jax.numpy a numpy standard, poi a lista di float
            "probabilities": np.asarray(dist.probabilities).astype(float).tolist()
        })
        
    with open(filepath, "wb") as f:
        pickle.dump(state, f)
    print(f"Model state safely saved to {filepath}")

if __name__ == "__main__":
    n = 484
    hw_min = 1
    hw_max = 2
    n_probs = 7
    maxiter = 50
    sweeps = 10
    top_n = 500
    sampling_fraction = 0.10 # Valutiamo solo il 10% di ~80.200 ks ad ogni sweep

    print(f"Loading spectrum array for {n} qubits...")
    target = TruncatedArraySpectrum(
        n=n, 
        pkl_path="expvals_IqpSimulator_dwave.pkl", 
        hw_min=hw_min, 
        hw_max=hw_max
    )
    
    print(f"Initializing Stochastic MMD (sampling {sampling_fraction*100:.1f}% of coefficients)...")
    mmd = SubsampledMMD(
        n=n, 
        sigma=0.25*n, 
        hw_min=hw_min, 
        hw_max=hw_max, 
        fraction=sampling_fraction
    )
    
    print(f"Active ks for optimization: {len(mmd.active_ks)}")
    print("Start optimization...", flush=True)
    t0 = time()
    
    trainer = SweepingLinearCombFitter(n_probs)
    result = trainer.fit(
        target,
        mmd,
        sweeps=sweeps,
        it_per_sweep=maxiter,
        verbose=True,
        fit_generators=True,
        save_history=True,
        top_n=top_n,
    )
    t1 = time()
    
    # Valutiamo l'errore finale sull'intero spettro visibile (opzionale, ma utile)
    full_mmd = SubsampledMMD(n=n, sigma=0.25*n, hw_min=hw_min, hw_max=hw_max, fraction=1.0)
    training_error = full_mmd(target, result)
    
    print(f"Done [{t1-t0:.2f}s] - Full MMD Error = {training_error:.3e}")
    
    model_filename = f"trained_combo_n{n}.pkl"
    save_safe_model(result, model_filename)
    
    plot_training_curve(trainer.history)