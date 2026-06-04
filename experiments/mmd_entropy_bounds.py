import numpy as np
import jax.numpy as jnp
from jax import random
import jax
from qfso.distributions.generate import generate_distribution_with_target_entropy
from qfso.models import match_first_order, mmd_exact
import matplotlib.pyplot as plt

def uniform_like(p):
    n = p.shape[0]
    return jnp.ones(n) / n

key = random.PRNGKey(0)
n_qubits = 10
n_entropy_points = 50
approx_distribution_type = "product" # either product or uniform

# 1. Setup e Masking
if approx_distribution_type=="product": sigmas = np.linspace(9 , 3*n_qubits, num=20)
if approx_distribution_type=="uniform": sigmas = np.linspace(0.5 , n_qubits/2, num=20)

min_entropy_global = np.log(2) * (n_qubits - np.max(sigmas) / 3)
max_entropy_global = (n_qubits - 3) * np.log(2)
entropies = np.linspace(min_entropy_global, max_entropy_global, num=n_entropy_points)

rescaled_entropies = n_qubits * np.log(2) - entropies
delta = rescaled_entropies[:, None] / (4 * sigmas[None, :])

if approx_distribution_type=="product":
    lower_bound = np.log(2) * (n_qubits - sigmas[None, :] / 3)
    upper_bound = (n_qubits - 3) * np.log(2)

if approx_distribution_type=="uniform":
    lower_bound = np.log(2) * (n_qubits - 2*sigmas[None, :])
    upper_bound = (n_qubits - 1) * np.log(2)

mask = (entropies[:, None] >= lower_bound) & (entropies[:, None] <= upper_bound)

# Inizializzazione array dei risultati
mmd_values = np.full_like(delta, np.nan)
mmd_values_unif = np.full_like(delta, np.nan)

# 2. Iterazione e calcolo MMD
for i, entropy in enumerate(entropies):
    valid_sigma_indices = np.where(mask[i])[0]
    
    if len(valid_sigma_indices) == 0:
        continue
        
    jnp_entropy = jnp.array(entropy)
    
    p = generate_distribution_with_target_entropy(2**n_qubits, jnp_entropy, key, 2)
    p_tilde = match_first_order(p)
    p_uniform = uniform_like(p)
    
    for j in valid_sigma_indices:
        jnp_sigma = jnp.array(sigmas[j])
        
        mmd_values[i, j] = mmd_exact(p, p_tilde, jnp_sigma)
        mmd_values_unif[i, j] = mmd_exact(p, p_uniform, jnp_sigma)

# 3. Flatten e filtering
delta_flat = delta.flatten()
mmd_flat = mmd_values.flatten()
mmd_unif_flat = mmd_values_unif.flatten()

if approx_distribution_type=="product": mask_plot = delta_flat <= 0.083 # 1/12
if approx_distribution_type=="uniform": mask_plot = delta_flat <= 0.5

d_filtered = delta_flat[mask_plot]
mmd_filtered = mmd_flat[mask_plot]
mmd_unif_filtered = mmd_unif_flat[mask_plot]

# 4. Plot
plt.figure(figsize=(10, 8))
if approx_distribution_type=="product": plt.scatter(d_filtered, mmd_filtered, alpha=0.5, label='Data (delta)')
if approx_distribution_type=="uniform": plt.scatter(d_filtered, mmd_unif_filtered, alpha=0.5, label='Uniform Baseline (delta)')

d_line = jnp.linspace(0, 0.1, 100)
# if approx_distribution_type=="product": plt.plot(d_line, 1152 * d_line**2, color='blue', linestyle='--', label=r'Theoretical Bound')
if approx_distribution_type=="product": plt.plot(d_line,  0.08 * jax.nn.sigmoid(130 * (d_line - 0.037)), color='blue', linestyle='--', label=r'Theoretical Bound')
if approx_distribution_type=="uniform": plt.plot(d_line, 8 * d_line, color='orange', linestyle='--', label=r'Theoretical Bound 2')

plt.yscale('log')
# plt.xscale('log')

plt.xlabel('Delta (Rescaled Entropy / (4 * Sigma))')
plt.ylabel('MMD^2')
plt.title('MMD^2 vs Delta (Filtered delta <= 0.083)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()