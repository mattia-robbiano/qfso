import jax
import optax

import jax.numpy as jnp
from model import _to_sparse

from copy import copy

#Parameters (optimizer)
lr = 0.05
epochs = 5

optimizer = optax.adam(lr)

def train(sampler, model, loss_fn, max_hw, verbose=True):

    # Jit the training step with hyperparameters

    @jax.jit
    def training_step(all_params, opt_state, ks, target_coeffs):

        # Sparsify
        sparse_ks = _to_sparse(ks, size=max_hw)

        # Compute loss and updates
        loss, grads = jax.value_and_grad(loss_fn)(all_params, sparse_ks, target_coeffs)
        updates, new_opt_state = optimizer.update(grads, opt_state, all_params)

        # Last row shall not be updated
        updates["params"] = updates["params"].at[-1,:].set(jnp.zeros(shape=(model.m,)))

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
    for e in range(epochs):
        current_sampler = copy(sampler)
        for b in range(sampler.num_batches):

            ks, target_coeffs = current_sampler.next_batch()
            loss, all_params, opt_state = training_step(all_params, opt_state, ks, target_coeffs)
            history.append(jnp.abs(loss))

        if verbose: 
            print(f"epoch: {e} loss: {loss:.3e}")
    
    return all_params, history

    