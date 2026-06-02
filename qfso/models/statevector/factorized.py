from jax import numpy as jnp
has_one = lambda x, bit: (x >> bit) & 1

def _compute_marginal(distribution, bit):
    """
    Compute the marginal distribution of the given distribution p over the specified bit.
    """
    m=0
    for i, p in enumerate(distribution):
        if has_one(i, bit): m += p
        
    return 1 - m, m

def match_first_order(distribution):
    """
    Find a factorized distribution that matches the first order moments of the given distribution p. 
    """
    assert len(distribution) & (len(distribution) - 1) == 0, "Distribution length must be a power of 2."
    n = len(distribution).bit_length() - 1
    
    marginals = [_compute_marginal(distribution, bit) for bit in range(n)]
    
    return jnp.array([jnp.prod(
        jnp.array([marginals[bit][has_one(i, bit)] for bit in range(n)])
        ) for i in range(len(distribution))])
