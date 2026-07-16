import jax
import jax.numpy as jnp

from model import LinCombModel, _to_sparse, _create_buffer
from iqp_sampler import IQPSampler

@jax.jit(static_argnames="n_ks")
def loss(fourier_matrix, latent_probs, sparse_ks, target_coeffs, n_ks):
    model_coeffs = _create_buffer(latent_probs, fourier_matrix, sparse_ks)
    return _mmd(model_coeffs, target_coeffs, n_ks)

@jax.jit(static_argnames="n_ks")
def _mmd(model_coeffs, target_coeffs, n_ks):
    diff = model_coeffs - target_coeffs
    return jnp.sum(diff*diff)/n_ks

@jax.jit(static_argnames="lr")
def _update_fourier(fourier_matrix, grad, lr):
    new_fourier = fourier_matrix[:-1,:] - lr*grad[:-1,:]
    return jnp.clip(new_fourier, min=-1, max=1)

if __name__ == "__main__":

    from qfso.models.approximate import IQPSampler

    dataset = "MNIST"
    n_qubits = 28*28
    pool_size = 100_000
    sigma = 7.2
    path = "/Users/crognale/Projects/qfso/data"
    batch_size = 1_000
    num_batches = 100

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
    model.fourier_matrix = model.fourier_matrix.at[:-1,:].set(jax.random.uniform(key, shape=(n_qubits,m), minval=-1, maxval=1))
    
    gradloss = jax.grad(loss)

    from time import time
    from copy import copy
    t = time()

    print("Done.\nStarting actual work")
    max_hw = 13
    lr = 100
    epochs = 100

    for e in range(epochs):
        tmp_sampler = copy(iqp_sampler)
        for _ in range(num_batches):
            ks, target_coeffs = tmp_sampler.next_batch()
            sparse_ks = _to_sparse(ks, size=max_hw)
            grad = gradloss(model.fourier_matrix, model.latent_probs, sparse_ks, target_coeffs, batch_size)
            #print(sparse_ks)
            #print(grad[sparse_ks[0,0]])
            model.fourier_matrix = model.fourier_matrix.at[:-1,:].set(_update_fourier(model.fourier_matrix, grad, lr))
        print(f"epoch:{e} \tloss:{loss(model.fourier_matrix, model.latent_probs, sparse_ks, target_coeffs, batch_size):.3e}")

    print(f"Done. Elapsed {time()-t:.3e}s")
