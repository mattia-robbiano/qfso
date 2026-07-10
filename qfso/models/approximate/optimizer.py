import jax
import jax.numpy as jnp
import optax
from functools import partial


@partial(jax.jit, static_argnums=(0, 3, 4))
def fit_stochastic(loss_fn, init_params, key, n_iters=10, lr=0.1):
    """JIT-compiled stochastic optimizer that samples batches natively on the GPU."""
    params = jnp.clip(jnp.asarray(init_params, dtype=jnp.float32), 0.0, 1.0)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        p, state, rng = carry
        # Split the key to generate a fresh random state for this iteration
        rng, subkey = jax.random.split(rng)
        
        # loss_fn now accepts the subkey to fetch the mini-batch
        grads = jax.grad(loss_fn)(p, subkey)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = jnp.clip(optax.apply_updates(p, updates), 0.0, 1.0)
        
        return (new_p, new_state, rng), None

    (final_params, _, _), _ = jax.lax.scan(step, (params, opt_state, key), None, length=n_iters)
    return final_params