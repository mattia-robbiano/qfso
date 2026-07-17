import os
import optax
from functools import partial
import jax
import jax.numpy as jnp
from time import time

from model import LinCombModel, _to_sparse, _create_buffer
from iqp_sampler import IQPSampler

@partial(jax.jit, static_argnums=4)
def loss(fourier_matrix, latent_probs, sparse_ks, target_coeffs, n_ks):
    model_coeffs = _create_buffer(latent_probs, fourier_matrix, sparse_ks)
    return _mmd(model_coeffs, target_coeffs, n_ks)

@partial(jax.jit, static_argnums=2)
def _mmd(model_coeffs, target_coeffs, n_ks):
    diff = model_coeffs - target_coeffs
    return jnp.sum(diff*diff)/n_ks

# Definizione dell'ottimizzatore globale. Adam con un lr standard parte molto meglio di SGD.
optimizer = optax.adam(learning_rate=0.5)

# n_ks è all'indice 4, opt_state (dinamico) è all'indice 5
@partial(jax.jit, static_argnums=4)
def train_step(fourier_matrix, latent_probs, sparse_ks, target_coeffs, n_ks, opt_state):
    # Calcolo congiunto di loss e gradienti
    batch_loss, grad = jax.value_and_grad(loss, argnums=0)(fourier_matrix, latent_probs, sparse_ks, target_coeffs, n_ks)
    
    # Passaggio a optax: ignoriamo l'ultima riga della matrice che deve restare costante a 1
    trainable_grad = grad[:-1, :]
    trainable_fourier = fourier_matrix[:-1, :]
    
    updates, new_opt_state = optimizer.update(trainable_grad, opt_state, trainable_fourier)
    new_trainable_fourier = optax.apply_updates(trainable_fourier, updates)
    
    # Riapplicazione del clip range [-1, 1]
    new_trainable_fourier = jnp.clip(new_trainable_fourier, min=-1.0, max=1.0)
    
    # Ricostruzione della matrice completa
    new_fourier_matrix = fourier_matrix.at[:-1, :].set(new_trainable_fourier)
    
    return new_fourier_matrix, new_opt_state, batch_loss

if __name__ == "__main__":

    dataset = "MNIST"
    n_qubits = 28*28
    pool_size = 1_000_000
    sigma = 7.1
    path = "../../../data/"
    
    # Modifica per aumentare l'esposizione: più batch per epoca
    batch_size = 10_000
    num_batches = 200 

    print("Doing preparation things...", end="", flush=True)

    iqp_sampler = IQPSampler(
        dataset=dataset,
        n_qubits=n_qubits,
        pool_size=pool_size,
        sigma=sigma,
        path=path,
        num_batches=num_batches,
        batch_size=batch_size,
    )

    m = 20

    model = LinCombModel(n_qubits, m)
    key = jax.random.PRNGKey(35)
    
    # Manteniamo l'inizializzazione strettamente positiva
    model.fourier_matrix = model.fourier_matrix.at[:-1,:].set(
        jax.random.uniform(key, shape=(n_qubits,m), minval=0.5, maxval=1.0)
    )

    # Inizializzazione dello stato interno di optax (momenti, varianze) solo sulla porzione addestrabile
    opt_state = optimizer.init(model.fourier_matrix[:-1, :])

    print("Done.\nStarting actual work")
    max_hw = 13
    epochs = 100

    t = time()
    for e in range(epochs):
        iqp_sampler.current_batch_idx = 0
        epoch_loss = 0.0
        
        for _ in range(num_batches):
            ks, target_coeffs = iqp_sampler.batch(to_jax=True)
            sparse_ks = _to_sparse(ks, max_hw)
            
            model.fourier_matrix, opt_state, batch_loss = train_step(
                model.fourier_matrix, 
                model.latent_probs, 
                sparse_ks, 
                target_coeffs, 
                batch_size,
                opt_state
            )
            epoch_loss += batch_loss
            
        print(f"epoch:{e} \tloss:{epoch_loss / num_batches:.3e}")

    print(f"Done. Elapsed {time()-t:.3e}s")