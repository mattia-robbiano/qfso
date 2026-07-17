import os
import gc
import numpy as np
import jax.numpy as jnp

class IQPSampler:
    """
    Extracts batches of operations and expectation values from a precomputed pool on disk.

    Args:
        dataset (str): Name of the dataset (e.g., "MNIST").
        n_qubits (int): Number of qubits in the IQP circuit.
        pool_size (int): Total number of samples available in the precomputed pool file.
        sigma (float): Standard deviation of the Bernoulli distribution used in the pool.
        num_batches (int): Number of batches to pre-extract. Default is 1.
        batch_size (int): Size of each batch to extract. Default is 1000.
        path (str): Path to the directory where the pool files are stored.
        seed (int): Random seed for reproducibility. Default is None.
    """
    def __init__(self, dataset: str, n_qubits: int, pool_size: int, sigma: float, 
                 num_batches: int, batch_size: int, path: str = "./", seed: int = None):

        self.dataset = dataset
        self.n_qubits = n_qubits
        self.pool_size = pool_size
        self.sigma = sigma
        self.path = path
        self.num_batches = num_batches
        self.batch_size = batch_size        
        self.current_batch_idx = 0

        self.ops_file = os.path.join(path, f'ops_pool_{dataset}_q{n_qubits}_s{sigma}.npy')
        self.expvals_file = os.path.join(path, f'expvals_pool_{dataset}_q{n_qubits}_s{sigma}.npy')

        if not (os.path.exists(self.ops_file) and os.path.exists(self.expvals_file)):
            raise FileNotFoundError(
                f"Pool files not found at {self.path}. Ensure you merged the chunks first."
            )

        # Open pool files with mmap_mode to avoid loading the master files into RAM
        mmap_ops = np.load(self.ops_file, mmap_mode='r')
        mmap_expvals = np.load(self.expvals_path, mmap_mode='r') if hasattr(self, 'expvals_path') else np.load(self.expvals_file, mmap_mode='r')

        rng = np.random.default_rng(seed)
        total_samples = num_batches * batch_size
        rand_idx = rng.integers(0, self.pool_size, size=total_samples)

        # Sort indices to force a single, sequential disk sweep (Lustre friendly)
        sort_perm = np.argsort(rand_idx)
        sorted_idx = rand_idx[sort_perm]
        all_ops = np.array(mmap_ops[sorted_idx])
        all_expvals = np.array(mmap_expvals[sorted_idx])

        # Un-sort
        inv_perm = np.argsort(sort_perm)
        all_ops = all_ops[inv_perm]
        all_expvals = all_expvals[inv_perm]
        self.batches_ops = np.split(all_ops, num_batches)
        self.batches_expvals = np.split(all_expvals, num_batches)

        # Close files
        del mmap_ops, mmap_expvals
        gc.collect()

    def batch(self, to_jax: bool = True):
        """
        Returns the next pre-extracted batch.

        Args:
            to_jax (bool): If True, returns JAX arrays. If False, returns NumPy arrays. Default is True.

        Returns:
            tuple: A tuple containing the operations and expectation values.
        """
        if self.current_batch_idx >= self.num_batches:
            raise IndexError("All pre-extracted batches have been consumed.")

        ops = self.batches_ops[self.current_batch_idx]
        expvals = self.batches_expvals[self.current_batch_idx]
        self.current_batch_idx += 1

        return (jnp.array(ops), jnp.array(expvals)) if to_jax else (ops, expvals)