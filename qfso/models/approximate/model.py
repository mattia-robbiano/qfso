from functools import partial
import jax
import jax.numpy as jnp

def _sample(key, m:int, n:int, latent_probs:jnp.ndarray, fourier_matrix:jnp.ndarray, n_samples:int):
    column_choices = jax.random.choice(key, jnp.arange(m), shape=(n_samples,), p=latent_probs)
    column, samples_to_be_taken = jnp.unique(column_choices, return_counts=True)

    p_matrix = 0.5*(1+fourier_matrix)

    samples_to_stack = []
    for c, ns in zip(column, samples_to_be_taken):
        samples_to_stack.append(jax.random.bernoulli(key, p_matrix[:,c], shape=(ns,n)))
    
    return jnp.vstack(samples_to_stack, dtype="int32")

@jax.jit
def _create_buffer(latent_probs:jnp.ndarray, fourier_matrix:jnp.ndarray, ks:jnp.ndarray):
    rows = fourier_matrix[ks]
    products = jnp.prod(rows, axis=1)
    return products@latent_probs

# 1 is the index of 'size' in the positional arguments
@partial(jax.jit,static_argnums=1)
def _to_sparse(arr, size):
    data_row_to_sparse = lambda a: jnp.nonzero(a, size=size, fill_value=-1)[0]
    return jax.vmap(data_row_to_sparse)(arr)

class LinCombModel():
    def __init__(self, n:int, m:int):
        self.n = n  
        self.m = m  

        self.latent_probs = jnp.full(shape=(m,), fill_value=1/m)
        self.fourier_matrix = jnp.zeros(shape=(n+1, m))
        self.fourier_matrix = self.fourier_matrix.at[-1,:].set(jnp.ones(shape=(m,)))

    def wh_spectrum(self, ks, max_hw):
        sparse_ks = _to_sparse(ks, max_hw)
        return _create_buffer(self.latent_probs, self.fourier_matrix, sparse_ks)

    def sample(self, n_samples, seed=42):
        key = jax.random.PRNGKey(seed)
        return _sample(key, self.m, self.n, self.latent_probs, self.fourier_matrix[:-1,:], n_samples)