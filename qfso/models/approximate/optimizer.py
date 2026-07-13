import jax
import jax.numpy as jnp
import optax
from functools import partial

@partial(jax.jit, static_argnums=(0, 6, 7))
def fit_stochastic(loss_fn, init_params, key, target_all, mask_all, filt_all, n_iters=10, lr=0.1):
    params = jnp.clip(jnp.asarray(init_params, dtype=jnp.float32), 0.0, 1.0)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        p, state, rng = carry
        rng, subkey = jax.random.split(rng)
        
        loss_val, grads = jax.value_and_grad(loss_fn)(p, subkey, target_all, mask_all, filt_all)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = jnp.clip(optax.apply_updates(p, updates), 0.0, 1.0)
        
        return (new_p, new_state, rng), loss_val

    (final_params, _, _), losses = jax.lax.scan(step, (params, opt_state, key), None, length=n_iters)
    return final_params, losses[-1]