import numpy as np
import jax.numpy as jnp
from jax import random
import jax
from qfso.distributions.generate import generate_distribution_with_target_entropy
from qfso.models import match_first_order, mmd_exact
import matplotlib.pyplot as plt

def uniform_like(p):
    return jnp.ones(p.shape[0]) / p.shape[0]

def compute_data(approx_distribution_type, n_qubits=10, n_entropy_points=50):
    key = random.PRNGKey(0)
    
    if approx_distribution_type == "product": 
        sigmas = np.linspace(9, 3 * n_qubits, num=20)
    else: 
        sigmas = np.linspace(0.5, n_qubits / 2, num=20)

    min_entropy_global = np.log(2) * (n_qubits - np.max(sigmas) / 3)
    max_entropy_global = (n_qubits - 3) * np.log(2)
    entropies = np.linspace(min_entropy_global, max_entropy_global, num=n_entropy_points)

    rescaled_entropies = n_qubits * np.log(2) - entropies
    delta = rescaled_entropies[:, None] / (4 * sigmas[None, :])

    if approx_distribution_type == "product":
        lower_bound = np.log(2) * (n_qubits - sigmas[None, :] / 3)
        upper_bound = (n_qubits - 3) * np.log(2)
    else:
        lower_bound = np.log(2) * (n_qubits - 2 * sigmas[None, :])
        upper_bound = (n_qubits - 1) * np.log(2)

    mask = (entropies[:, None] >= lower_bound) & (entropies[:, None] <= upper_bound)
    mmd_values = np.full_like(delta, np.nan)
    mmd_values_unif = np.full_like(delta, np.nan)

    for i, entropy in enumerate(entropies):
        valid_sigma_indices = np.where(mask[i])[0]
        if len(valid_sigma_indices) == 0: continue
            
        p = generate_distribution_with_target_entropy(2**n_qubits, jnp.array(entropy), key, 2)
        p_tilde = match_first_order(p)
        p_uniform = uniform_like(p)
        
        for j in valid_sigma_indices:
            jnp_sigma = jnp.array(sigmas[j])
            mmd_values[i, j] = mmd_exact(p, p_tilde, jnp_sigma)
            mmd_values_unif[i, j] = mmd_exact(p, p_uniform, jnp_sigma)

    delta_flat = delta.flatten()
    mask_plot = delta_flat <= (0.083 if approx_distribution_type == "product" else 0.5)
    
    return delta_flat[mask_plot], mmd_values.flatten()[mask_plot], mmd_values_unif.flatten()[mask_plot]

# 1. Generazione dati per entrambi i tipi
d_prod, mmd_prod, _ = compute_data("product")
d_unif, _, mmd_unif = compute_data("uniform")

# 2. Plot unico
plt.figure(figsize=(10, 8))

# Scatter punti
plt.scatter(d_prod, mmd_prod, alpha=0.5, color='blue', label='Product Data (delta)')
plt.scatter(d_unif, mmd_unif, alpha=0.5, color='orange', label='Uniform Baseline (delta)')

# Linee teoriche
d_line_prod = jnp.linspace(0, 0.1, 100)
d_line_unif = jnp.linspace(0, 0.5, 100)

plt.plot(d_line_prod, 1152 * (d_line_prod ** 2), 
         color='darkblue', linestyle='--', label='Theoretical Bound (Product)')
plt.plot(d_line_unif, 8 * d_line_unif, 
         color='darkorange', linestyle='--', label='Theoretical Bound (Uniform)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Delta (Rescaled Entropy / (4 * Sigma))')
plt.ylabel('MMD^2')
plt.title('Comparison: Product vs Uniform MMD^2')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()