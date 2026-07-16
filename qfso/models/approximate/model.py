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

    # selects the row necessary for each frequency
    rows = fourier_matrix[ks]
    # makes the product of fourier coefficients. Since the last row is always
    # ones, this accounts for changing shapes, as all -1 entries in ks are
    # virtually ignored
    products = jnp.prod(rows, axis=1)
    # weights the single probability coefficients by the latent probabilities
    return products@latent_probs

@jax.jit(static_argnames="size")
def _to_sparse(arr, size):

    # single row operation. If the number of non-zero bits are less than size,
    # fill with reference to the last row, which by construnction is only ones
    data_row_to_sparse = lambda a: jnp.nonzero(a, size=size, fill_value=-1)[0]
    # vectorized operation for all rows
    return jax.vmap(data_row_to_sparse)(arr)

class LinCombModel():

    def __init__(self, n:int, m:int):
        
        self.n = n  #number of bits
        self.m = m  #number of probability distributions in the approximation

        #relative weight of the factorized distribution
        self.latent_probs = jnp.full(shape=(m,), fill_value=1/m)
        #spectrums of the factorized distributions, plus an all-1 row in the end
        self.fourier_matrix = jnp.zeros(shape=(n+1, m))
        self.fourier_matrix = self.fourier_matrix.at[-1,:].set(jnp.ones(shape=(m,)))

    def wh_spectrum(self, ks, max_hw):
        sparse_ks = _to_sparse(ks, size = max_hw)
        return _create_buffer(self.latent_probs, self.fourier_matrix, sparse_ks)

    def sample(self, n_samples, seed=42):
        key = jax.random.PRNGKey(seed)
        return _sample(key, self.m, self.n, self.latent_probs, self.fourier_matrix[:-1,:], n_samples)

if __name__ == "__main__":

    n = 135
    m = 306

    model = LinCombModel(n, m)

    key = jax.random.PRNGKey(35)
    model.fourier_matrix = model.fourier_matrix.at[:-1,:].set(jax.random.uniform(key, shape=(n,m), minval=-1, maxval=1))
    print(model.fourier_matrix)

    import itertools
    ks = []
    for c in itertools.combinations(range(n), 2):
        k = jnp.zeros(n)
        for ci in c:
            k = k.at[ci].set(1)
        ks.append(k)

    ks_dense = jnp.array(ks)
    print(len(ks_dense))
    
    from time import time
    t = time()
    print(model.wh_spectrum(ks_dense, 2))
    print(f"{time()-t:.3e}s")

    print(model.fourier_matrix)
    rows = model.fourier_matrix[jnp.array([[1,-1,-1], [0,3,-1]])]
    print(rows)
    print(jnp.prod(rows, axis=1))