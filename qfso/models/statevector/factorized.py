from jax import numpy as jnp
from jax import jit


def has_one(x, bit):
    return (x >> bit) & 1

@jit
def _compute_marginal(distribution, bit):
    """
    Compute the marginal distribution of the given distribution p over the specified bit.
    """
    indices = jnp.arange(distribution.shape[0])
    m = jnp.sum(jnp.where(has_one(indices, bit) == 1, distribution, 0))
    return jnp.array([1 - m, m])

@jit
def match_first_order(distribution):
    """
    Find a factorized distribution that matches the first order moments of the given distribution p. 
    """
    n_states = distribution.shape[0]
    n = n_states.bit_length() - 1

    indices = jnp.arange(n_states)
    bits = jnp.arange(n)
    marginals = jnp.stack([_compute_marginal(distribution, bit) for bit in range(n)])
    bit_values = ((indices[:, None] >> bits[None, :]) & 1).astype(jnp.int32)
    probs = jnp.take_along_axis(marginals[None, :, :], bit_values[:, :, None], axis=2).squeeze(-1)
    return jnp.prod(probs, axis=1)
