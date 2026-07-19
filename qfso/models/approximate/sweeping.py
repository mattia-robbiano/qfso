import jax
import optax

import jax.numpy as jnp
from model import _to_sparse

from copy import copy

#Parameters (optimizer)
lr = 0.05
sweeps = 5

optimizer = optax.adam(lr)

def train(sampler, model, loss_fn, max_hw, verbose=True):

    # Jit the training step with hyperparameters

    @jax.jit(static_argnames=("prob_num", "only_latent"))
    def training_step(all_params, opt_state, ks, target_coeffs, prob_num, only_latent=False):

        # Sparsify
        sparse_ks = _to_sparse(ks, size=max_hw)

        # Compute loss and updates
        loss, grads = jax.value_and_grad(loss_fn)(all_params, sparse_ks, target_coeffs)
        updates, new_opt_state = optimizer.update(grads, opt_state, all_params)

        # Select only column to update
        if only_latent:
            updates["params"] = updates["params"].at[:,:].set(jnp.zeros_like(updates["params"]))
        else:
            updates["params"] = updates["params"].at[-1,:].set(jnp.zeros(shape=(model.m,)))
            if prob_num > 0:
                updates["params"] = updates["params"].at[:-1,:prob_num-1].set(jnp.zeros(shape=(model.n,prob_num-1)))
            if prob_num < model.m-1:
                updates["params"] = updates["params"].at[:-1,prob_num+1:].set(jnp.zeros(shape=(model.n,model.m-prob_num-1)))
            updates["latent_params"] = updates["latent_params"].at[:].set(jnp.zeros(shape=(model.m,)))

        # Update the rest
        updated_params = optax.apply_updates(all_params, updates)

        return loss, updated_params, new_opt_state
    
    # Build the optimizer
    
    all_params = {
        "params":model.params, 
        "latent_params":model.latent_params,
    }
    opt_state = optimizer.init(all_params)

    # Main loop
    
    history = []
    for s in range(sweeps):
        for prob in range(model.m):
            current_sampler = copy(sampler)
            for b in range(sampler.num_batches):

                ks, target_coeffs = current_sampler.next_batch()
                loss, all_params, opt_state = training_step(all_params, opt_state, ks, target_coeffs, prob, False)
                history.append(jnp.abs(loss))

        current_sampler = copy(sampler)
        for b in range(sampler.num_batches):
            ks, target_coeffs = current_sampler.next_batch()
            loss, all_params, opt_state = training_step(all_params, opt_state, ks, target_coeffs, -1, True)
            history.append(jnp.abs(loss))

        if verbose: 
            print(f"sweep: {s} loss: {loss:.3e}")
    
    return all_params, history