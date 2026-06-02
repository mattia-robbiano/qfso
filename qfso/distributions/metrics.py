import jax
import jax.numpy as jnp

@jax.jit
def mmd_qubits(p: jnp.ndarray, q: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Exact MMD between two distributions over n sites using an RBF kernel.
    
    Args:
        p, q: 1D arrays of probabilities. Shape: (n,)
        sigma: RBF kernel bandwidth.
    """
    N = p.shape[0]
    
    domain = jnp.arange(N, dtype=jnp.float32)
    
    # (i - j)^2 for all pairs of states
    dist_sq = jnp.square(domain[:, None] - domain[None, :])
    
    K = jnp.exp(-dist_sq / (2.0 * sigma**2))
    
    diff = p - q
    return jnp.dot(diff, jnp.dot(K, diff))