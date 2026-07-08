import jax
import jax.numpy as jnp
import optax
from functools import partial

@partial(jax.jit, static_argnums=(0, 2, 3))
def fit(loss_fn, init_params, n_iters=10, lr=0.1):
    params = jnp.clip(jnp.asarray(init_params, dtype=jnp.float32), 0.0, 1.0)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        p, state = carry
        grads = jax.grad(loss_fn)(p)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = jnp.clip(optax.apply_updates(p, updates), 0.0, 1.0)
        return (new_p, new_state), None

    (final_params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=n_iters)
    return final_params
