import os
import pickle
import time
import yaml
import numpy as np
import jax
import jax.numpy as jnp

from iqpopt import IqpSimulator
from iqpopt.utils import local_gates

class IQPSampler:
    """
    Generates batches of operations and expectation values on-the-fly 
    from the Fourier spectrum of an IQP circuit.

    Args:
        dataset (str): Name of the dataset (e.g., "MNIST").
        n_qubits (int): Number of qubits in the IQP circuit.
        sigma (float): Standard deviation of the Bernoulli distribution used to sample Z strings.
        path (str): Path to the directory where hyperparameters and parameters are stored.
        num_batches (int): Maximum number of batches to generate. Default is 1.
        batch_size (int): Size of each batch to generate. Default is 1000.
        seed (int): Random seed for reproducibility. Default is None.
    """
    def __init__(self, dataset: str, n_qubits: int, sigma: float, path: str = "./", seed: int = None):

        self.dataset = dataset
        self.n_qubits = n_qubits
        self.sigma = sigma
        self.path = path
        self.internal_max_batch = 1000
        self.p_sigma = (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        
        init_seed = seed if seed is not None else time.time_ns()
        self.key = jax.random.PRNGKey(init_seed)


        with open(os.path.join(path, 'best_hyperparameters.yaml'), 'r') as f:
            best_hyperparams = yaml.safe_load(f)
        with open(os.path.join(path, f'params_IqpSimulator_{dataset}.pkl'), 'rb') as f:
            self.params_iqp = pickle.load(f)
        gate_fn = globals()[best_hyperparams['IqpSimulator'][dataset]['gates_config']['name']]
        gates = gate_fn(**best_hyperparams['IqpSimulator'][dataset]['gates_config']['kwargs'])
        self.model = IqpSimulator(**best_hyperparams['IqpSimulator'][dataset]['model_config'], gates=gates)

    def batch(self, batch_size: int, internal_max_batch: int = 10_000):
        """
        Generates and returns the next batch of operations and expectation values on-the-fly.

        Args:
            to_jax (bool): If True, returns JAX arrays. If False, returns NumPy arrays. Default is True.

        Returns:
            tuple: A tuple containing the operations and expectation values.
        """
        self.batch_size = batch_size        

        self.key, sample_key, model_key = jax.random.split(self.key, 3)

        op_chunk = jax.random.bernoulli(
            sample_key, p=self.p_sigma, shape=(self.batch_size, self.n_qubits)
        )

        e, _ = self.model.op_expval(
            params=self.params_iqp, 
            ops=op_chunk, 
            n_samples=100,  
            key=model_key,
            max_batch_ops=internal_max_batch
        )

        return op_chunk, e


initial_time = time.time()
sampler = IQPSampler(
    dataset="MNIST",
    n_qubits=784,
    sigma=7.1,
    path="./",
    )
final_time = time.time()
print(f"Time taken to initialize IQPSampler: {final_time - initial_time} seconds")

initial_time = time.time()
ops, exp = sampler.batch(batch_size=10_000, internal_max_batch=1000)
final_time = time.time()

print(f"Time taken to generate batch: {final_time - initial_time} seconds")

# initial_time = time.time()
# np.savez_compressed("iqp_sampler_batch.npz", ops=ops, expvals=exp)
# final_time = time.time()
# print(f"Time taken to save batch: {final_time - initial_time} seconds")