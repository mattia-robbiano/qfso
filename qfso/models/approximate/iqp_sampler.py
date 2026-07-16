import os
import gc
import pickle
import yaml
import numpy as np
import jax
import jax.numpy as jnp

from iqpopt import IqpSimulator
from iqpopt.utils import local_gates

PRECISION_DTYPE = jnp.float32  
NP_PRECISION_DTYPE = np.float32 

class IQPSampler:
    """
    Computes a pool of coefficients from the whole Fourier spectrum of an IQP circuit,
    using IQPOpt Z string efficient calculation.

    Args:
        dataset (str): Name of the dataset (e.g., "MNIST").
        n_qubits (int): Number of qubits in the IQP circuit.
        pool_size (int): Number of samples to generate in the pool.
        sigma (float): Standard deviation of the Bernoulli distribution used to sample Z strings.
        path (str): Path to the directory where the pool files will be stored.
        gen_batch_size (int): Batch size for pool generation. Default is 5000.
        num_batches (int): Number of batches to extract from the pool. Default is 1.
        batch_size (int): Size of each batch to extract from the pool. Default is 1000.
        seed (int): Random seed for reproducibility. Default is None.
    """
    def __init__(self, dataset: str, n_qubits: int, pool_size: int, sigma: float, 
                 path: str = "./", gen_batch_size: int = 5000,
                 num_batches: int = 1, batch_size: int = 1000, seed: int = None):
        """
        Computes a pool of coefficients from the whole Fourier spectrum of an IQP circuit,
        using IQPOpt Z string efficient calculation.

        Args:
            dataset (str): Name of the dataset (e.g., "MNIST").
            n_qubits (int): Number of qubits in the IQP circuit.
            pool_size (int): Number of samples to generate in the pool.
            sigma (float): Standard deviation of the Bernoulli distribution used to sample Z strings.
            path (str): Path to the directory where the pool files will be stored.
            gen_batch_size (int): Batch size for pool generation. Default is 5000.
            num_batches (int): Number of batches to extract from the pool. Default is 1.
            batch_size (int): Size of each batch to extract from the pool. Default is 1000.
            seed (int): Random seed for reproducibility. Default is None.
        """

        self.dataset = dataset
        self.n_qubits = n_qubits
        self.pool_size = pool_size
        self.sigma = sigma
        self.path = path
        self.current_batch_idx = 0
        self.num_batches = num_batches
        
        self.ops_file = os.path.join(path, f'ops_pool_{dataset}_q{n_qubits}_s{sigma}.npy')
        self.expvals_file = os.path.join(path, f'expvals_pool_{dataset}_q{n_qubits}_s{sigma}.npy')

        if not (os.path.exists(self.ops_file) and os.path.exists(self.expvals_file)):
            self._generate_pool(gen_batch_size)
            
        mmap_ops = np.load(self.ops_file, mmap_mode='r')
        mmap_expvals = np.load(self.expvals_file, mmap_mode='r')
        
        rng = np.random.default_rng(seed)
        total_samples = num_batches * batch_size
        
        # Generate all random indices at once
        rand_idx = rng.integers(0, self.pool_size, size=total_samples)
        
        # Sort indices to force a single, sequential disk sweep (Lustre friendly)
        sort_perm = np.argsort(rand_idx)
        sorted_idx = rand_idx[sort_perm]
        
        # Single-shot disk read
        all_ops = np.array(mmap_ops[sorted_idx])
        all_expvals = np.array(mmap_expvals[sorted_idx])
        
        # Un-sort arrays in RAM to restore true stochastic distribution across batches
        inv_perm = np.argsort(sort_perm)
        all_ops = all_ops[inv_perm]
        all_expvals = all_expvals[inv_perm]
        
        # Split the large RAM arrays into sub-batches
        self.batches_ops = np.split(all_ops, num_batches)
        self.batches_expvals = np.split(all_expvals, num_batches)

        # flushing ram
        del mmap_ops, mmap_expvals
        gc.collect()

    def next_batch(self, to_jax: bool = True):
        """
        Returns the next batch of operations and expectation values 
        (batches are given one by one for convenience, but they are pre-extracted from the pool).
        Args:
            to_jax (bool): If True, returns the batch as JAX arrays. If False, returns them as NumPy arrays. Default is True.
        Returns:
            tuple: A tuple containing the operations and expectation values for the next batch.
        """
        if self.current_batch_idx >= self.num_batches:
            raise IndexError("All pre-extracted batches have been consumed.")
            
        ops = self.batches_ops[self.current_batch_idx]
        expvals = self.batches_expvals[self.current_batch_idx]
        self.current_batch_idx += 1
        
        return (jnp.array(ops), jnp.array(expvals)) if to_jax else (ops, expvals)

    def _generate_pool(self, batch_size: int):
        backend = jax.default_backend()
        internal_max_batch = 1000 if backend == "gpu" else 10000

        with open(os.path.join(self.path, 'best_hyperparameters.yaml'), 'r') as f:
            best_hyperparams = yaml.safe_load(f)
        with open(os.path.join(self.path, f'params_IqpSimulator_{self.dataset}.pkl'), 'rb') as f:
            params_iqp = pickle.load(f)

        gate_fn = globals()[best_hyperparams['IqpSimulator'][self.dataset]['gates_config']['name']]
        gates = gate_fn(**best_hyperparams['IqpSimulator'][self.dataset]['gates_config']['kwargs'])
        model = IqpSimulator(**best_hyperparams['IqpSimulator'][self.dataset]['model_config'], gates=gates)

        p_sigma = (1.0 - jnp.exp(-1.0 / (2.0 * self.sigma**2))) / 2.0
        
        ops_mmap = np.lib.format.open_memmap(self.ops_file, mode='w+', dtype=np.int32, shape=(self.pool_size, self.n_qubits))
        expvals_mmap = np.lib.format.open_memmap(self.expvals_file, mode='w+', dtype=NP_PRECISION_DTYPE, shape=(self.pool_size,))

        key = jax.random.PRNGKey(33)
        count = 0
        total_batches = int(np.ceil(self.pool_size / batch_size))

        for _ in range(total_batches):
            current_batch_size = min(batch_size, self.pool_size - count)
            key, sample_key, model_key = jax.random.split(key, 3)
            
            op_chunk = jax.random.bernoulli(
                sample_key, p=p_sigma, shape=(current_batch_size, self.n_qubits)
            ).astype(jnp.int32)
            
            e, _ = model.op_expval(
                params=params_iqp, 
                ops=op_chunk, 
                n_samples=100,  
                key=model_key,
                max_batch_ops=internal_max_batch
            )
            
            ops_mmap[count:count + current_batch_size] = np.array(jax.device_get(op_chunk))
            expvals_mmap[count:count + current_batch_size] = np.array(jax.device_get(e.astype(PRECISION_DTYPE)))
            count += current_batch_size

        ops_mmap.flush()
        expvals_mmap.flush()