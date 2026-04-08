import jax
import jax.numpy as jnp


@jax.jit
def median_heuristic(X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the median-heuristic estimate used for RBF bandwidth selection.
    """
    diffSq = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    distances = jnp.sqrt(diffSq)
    return jnp.median(distances)


def sigma_spectrum(X: jnp.ndarray, n_sigmas: int = 1) -> list[float]:
    """
    Return sigma values between the median heuristic and sigma_2.
    """
    med = median_heuristic(X[:2000])
    sigma_2 = jnp.sqrt(-1 / (2 * jnp.log(1 - 4 / X.shape[-1])))
    lower_bound = jnp.maximum(sigma_2, 1e-4)
    upper_bound = jnp.maximum(med, 1e-4)
    return [float(s) for s in jnp.linspace(lower_bound, upper_bound, n_sigmas)]


def sigma_heuristic(X: jnp.ndarray) -> float:
    """
    Return a stable single-sigma heuristic for training defaults.
    """
    med = median_heuristic(X[:2000])
    return float(jnp.maximum(med, 1e-4))


__all__ = ["median_heuristic", "sigma_spectrum", "sigma_heuristic"]
