import jax
import jax.numpy as jnp

@jax.jit
def _params_to_probs(params, latent_params):
    probs = jax.nn.sigmoid(params)
    latent_probs = jax.nn.sigmoid(latent_params)
    latent_probs = latent_probs/jnp.sum(latent_probs)
    return probs, latent_probs

@jax.jit
def _params_to_fourier(params, latent_params):
    fourier_matrix = 1-2*jax.nn.sigmoid(params)
    latent_probs = jax.nn.sigmoid(latent_params)
    latent_probs = latent_probs/jnp.sum(latent_probs)
    return fourier_matrix, latent_probs

def _sample(key, m:int, n:int, latent_probs:jnp.ndarray, p_matrix:jnp.ndarray, n_samples:int):
    
    column_choices = jax.random.choice(key, jnp.arange(m), shape=(n_samples,), p=latent_probs)
    column, samples_to_be_taken = jnp.unique(column_choices, return_counts=True)

    samples_to_stack = []
    for c, ns in zip(column, samples_to_be_taken):
        samples_to_stack.append(jax.random.bernoulli(key, p_matrix[:,c], shape=(ns,n)))
    
    return jnp.vstack(samples_to_stack, dtype="int32")

@jax.jit
def _compute_coefficients(latent_probs:jnp.ndarray, fourier_matrix:jnp.ndarray, ks:jnp.ndarray):

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

LARGE_NUMBER = -100.0

class LinCombModel():

    def __init__(self, n:int, m:int):
        
        self.n = n  #number of bits
        self.m = m  #number of probability distributions in the approximation

        #relative weight of the factorized distribution
        self.latent_params = jnp.full(shape=(m,), fill_value=1/m)
        #spectrums of the factorized distributions, plus an all-1 row in the end
        self.params = jnp.zeros(shape=(n+1, m))
        self.params = self.params.at[-1,:].set(LARGE_NUMBER*jnp.ones(shape=(m,)))

    def randomize_params(self, key):
        self.params = self.params.at[:-1,:].set(jax.random.uniform(key, shape=(self.n,self.m), minval=-10, maxval=10))

    def sample(self, n_samples, seed=42):
        key = jax.random.PRNGKey(seed)
        p_matrix, latent_probs = _params_to_probs(self.params, self.latent_params)
        return _sample(key, self.m, self.n, latent_probs, p_matrix[:-1,:], n_samples)

if __name__ == "__main__":

    n = 3
    m = 2
    max_hw = 20

    model = LinCombModel(n, m)

    key = jax.random.PRNGKey(1)
    model.randomize_params(key)

    fourier, latent = _params_to_fourier(model.params, model.latent_params)
    print(fourier, latent)
    
    #probs, latent = _params_to_probs(model.params, model.latent_params)
    #print(probs, latent)
    
    ks = jnp.array([[0,0,1], [1,0,0], [0,1,1]])
    sparse_ks = _to_sparse(ks, max_hw)
    coeffs = _compute_coefficients(latent, fourier, sparse_ks)
    print(sparse_ks, coeffs)

