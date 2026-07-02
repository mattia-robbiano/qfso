from tqdm import tqdm

import numpy as np
import jax.random as random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qfso.models.iqp import IQPTensorNetwork, local_gates, sigma_heuristic, setup_training
from qfso.utils import convert_to_jnp_ndarray

# hyperparameters
nqubits = 36
sigma = 0.1 * nqubits
# sigma = sigma_heuristic(X=ground_truth)
key = random.PRNGKey(42)


# circuit initialization
gate_list = local_gates(nqubits, max_weight=2)
iqp = IQPTensorNetwork(nqubits=nqubits, interactions=gate_list)
params = random.uniform(key, shape=(len(gate_list),), minval=0, maxval=2*jnp.pi)
circuit = iqp.build_circuit(parameters=params)

generators = convert_to_jnp_ndarray(gate_list, n_qubits=nqubits)
ground_truth = jnp.asarray(np.load("./datasets/ising_L6_T2.4_h0.08.npy"))

opt_state, train_step = setup_training(
    init_params=params, 
    generators=generators, 
    ground_truth=ground_truth, 
    sigma=sigma, 
    n_ops=1000, 
    n_samples=1000, 
    lr=0.002,
)

epochs = 1000
loss_history = []
for _ in tqdm(range(epochs), desc="Training", leave=False):
    key, subkey = random.split(key)
    params, opt_state, loss_val = train_step(params, opt_state, subkey)
    loss_history.append(loss_val)


loss_history_np = np.asarray(jnp.stack(loss_history))
params_np = np.asarray(params)
np.savez(
    "training_results.npz",
    loss_history=loss_history_np,
    params=params_np,
    epochs=epochs,
    nqubits=nqubits,
    sigma=float(sigma),
)


plt.figure(figsize=(7, 4))
plt.plot(loss_history_np)
plt.xlabel("Epoch")
plt.ylabel("MMD Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()