import jax
import optax

import jax.numpy as jnp

from iqp_sampler import IQPSampler
from model import LinCombModel, _compute_coefficients, _to_sparse, _params_to_fourier

from time import time
from copy import copy

@jax.jit
def loss_fn(all_params, sparse_ks, target_coeffs):
    fourier_matrix, latent_probs = _params_to_fourier(all_params["params"], all_params["latent_params"]) 
    model_coeffs = _compute_coefficients(latent_probs, fourier_matrix, sparse_ks)
    return _mmd_biased(model_coeffs, target_coeffs)
    #return _mmd_scalar_prod(model_coeffs, target_coeffs)

@jax.jit
def _mmd_biased(model_coeffs, target_coeffs):
    diff = model_coeffs - target_coeffs
    return jnp.sum(diff*diff)/batch_size

@jax.jit
def _mmd_scalar_prod(model_coeffs, target_coeffs):
    return -jnp.sum(model_coeffs*target_coeffs)/batch_size

if __name__ == "__main__":

    # Parameters (data)
    dataset = "MNIST"
    n_qubits = 28*28
    pool_size = 1_000_000
    sigma = 7.1
    path = "/Users/crognale/Projects/qfso/data"
    batch_size = 10_000
    num_batches = 100
    max_hw = 20
    
    # Parameters (model)
    n_probs = 10
    randomization_seed = 42

    t0=time()
    print("Setting up data...", flush=True, end="")

    # Data preparation
    iqp_sampler = IQPSampler(
        dataset=dataset,
        n_qubits=n_qubits,
        pool_size=pool_size,
        sigma=sigma,
        path=path,
        num_batches=num_batches,
        batch_size=batch_size,
    )
    print(f"Done. [{time()-t0:.3e}s]")

    t0=time()
    print("Setting up model...", flush=True, end="")

    # Build the model
    model = LinCombModel(n_qubits, n_probs)
    key = jax.random.PRNGKey(randomization_seed)
    model.randomize_params(key)

    print(f"Done. [{time()-t0:.3e}s]")

    t0=time()
    print("Training...", flush=True)

    import synchronous
    import sweeping

    method = sweeping

    all_params, history = method.train(iqp_sampler, model, loss_fn, max_hw)

    print(f"Done. [{time()-t0:.3e}s]")

    # Get one sample
    model.params = all_params["params"]
    model.latent_params = all_params["latent_params"]

    import numpy as np
    import matplotlib.pyplot as plt

    samples= np.array(model.sample(24))
    samples = samples.reshape(24,28,28)

    fig, axes = plt.subplots(4, 6)
    axes = axes.flatten()

    for i, sample in enumerate(samples):
        ax = axes[i]
        ax.imshow(sample, cmap='gray', interpolation='nearest')
        ax.axis('off')

    fig.suptitle(
        rf"N={n_qubits}, M={n_probs}, $\sigma$={sigma}, Pool: {pool_size} random frequencies"\
        f"\n (Final IQP MMD^2 = {history[-1]:.3e})",
        fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.title("Training curve")
    plt.semilogy(history)
    plt.show()