import pickle
from time import time
import numpy as np

from qfso.models.approximate import (
    LinCombApproximation, 
    SweepingLinearCombFitter,
)
from qfso.models.approximate.probability.spectrum import TruncatedArraySpectrum
from qfso.models.approximate.metrics import SubsampledMMD

def plot_training_curve(history, filename):
    import matplotlib.pyplot as plt
    plt.semilogy(range(len(history)), history)
    plt.xlabel("sweeps")
    plt.ylabel("MMD^2")
    plt.title("Training curve (Stochastic MMD)")
    plt.savefig("filename")

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

def evaluate_full_mmd(target, approx, mmd):
    """
    Calcola la MMD totale sull'intera popolazione dei coefficienti 
    usando la modalità eager di JAX (sicuro per la RAM).
    """
    import jax.numpy as jnp

    all_ks = mmd.all_ks
    len_hw1 = target.n
    
    target_hat = target.walsh_hadamard_spectrum(ks=all_ks)
    approx_hat = approx.walsh_hadamard_spectrum(ks=all_ks)
    
    mse1 = jnp.mean((target_hat[:len_hw1] - approx_hat[:len_hw1]) ** 2)
    mse2 = jnp.mean((target_hat[len_hw1:] - approx_hat[len_hw1:]) ** 2)
    
    filt_np = np.asarray(mmd.all_weights)
    total_mmd = filt_np[0] * mse1 + filt_np[1] * mse2 
    
    return float(total_mmd)

if __name__ == "__main__":
    n = 484
    hw_min = 1
    hw_max = 2
    
    n_probs = 15
    maxiter = 100
    sweeps = 10
    sampling_fraction = 0.05 # sotto 0.0005 le loss sono tutte NaN occhio!!

    sigma = n * 0.25 # tcdq values: 7.8, 6.1, 3.9

    print(f"Loading spectrum array for {n} qubits...")
    target = TruncatedArraySpectrum(
            n=n, 
            pkl_path="expvals_IqpSimulator_dwave.pkl", 
            hw_min=hw_min,
            hw_max=hw_max,
            file_hw_min=1,
            file_hw_max=2        
        )
    
    print(f"Initializing Stochastic MMD (sampling {sampling_fraction*100:.1f}% of coefficients)...")
    mmd = SubsampledMMD(
        n=n, 
        sigma=sigma,
        hw_min=hw_min, 
        hw_max=hw_max, 
        fraction=sampling_fraction
    )
    
    print(f"Active ks for optimization: {len(mmd.all_ks)}")
    print("Start optimization...", flush=True)
    t0 = time()
    
    trainer = SweepingLinearCombFitter(n_probs)
    result = trainer.fit(
        target,
        mmd,
        sweeps=sweeps,
        it_per_sweep=maxiter,
        verbose=True,
        fit_generators=False,
        save_history=True,
        top_n=None
    )

    t1 = time()
    
    model_filename = f"trained_combo_n{n}.pkl"
    save_safe_model(result, model_filename)

    history_filename = f"trained_combo_n{n}.png"
    plot_training_curve(trainer.history, history_filename)
    
    full_mmd = SubsampledMMD(n=n, sigma=sigma, hw_min=hw_min, hw_max=hw_max, fraction=1.0)
    training_error = evaluate_full_mmd(target, result, mmd)
    