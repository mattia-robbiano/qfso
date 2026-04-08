import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from quimb import pauli
from quimb.tensor import Circuit
from typing import List, Optional


def expvals_contraction(circ: Circuit, sites):
    """
    Compute expectation values for 1- and 2-local Pauli-Z operators.
    """
    expvals = []
    for op in sites:
        if len(op) == 1:
            expvals.append(circ.local_expectation(G=pauli("Z"), where=op[0]).real)
        elif len(op) == 2:
            expvals.append(circ.local_expectation(G=pauli("Z") & pauli("Z"), where=op).real)
        else:
            raise ValueError("Only 1 or 2-site operators are supported.")

    return jnp.array(expvals)


def expvals_sampling(circuit: Circuit, ops: jnp.ndarray, n_samples: int, seed: int = None) -> tuple:
    """
    Estimate expectation values by sampling the final state of a quimb circuit.
    """
    if isinstance(ops, list):
        ops_binary = np.zeros((len(ops), circuit.N), dtype=int)
        for i, sites in enumerate(ops):
            for s in sites:
                ops_binary[i, s] = 1
        ops = jnp.array(ops_binary)

    samples_raw = circuit.sample(
        n_samples,
        qubits=None,
        order=None,
        group_size=10,
        max_marginal_storage=2**20,
        seed=seed,
        optimize="auto-hq",
        backend="jax",
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-06,
        simplify_equalize_norms=True,
    )
    samples_mat = jnp.array([[int(bit) for bit in s] for s in samples_raw])

    parity = (ops @ samples_mat.T) % 2
    eigenvalues = 1 - 2 * parity

    expvals = jnp.mean(eigenvalues, axis=1)
    std_devs = jnp.std(eigenvalues, axis=1, ddof=1) / jnp.sqrt(n_samples)

    return expvals, std_devs


def expvals_mc(
    params: jnp.ndarray,
    ops: jnp.ndarray,
    generators: jnp.ndarray,
    n_samples: int,
    key: Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimate expectation values of Pauli-Z operators for an IQP circuit.
    """
    n_qubits = generators.shape[1]

    samples = jax.random.randint(key, shape=(n_samples, n_qubits), minval=0, maxval=2)
    ops_gen = (ops @ generators.T) % 2
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
    par_ops_gates = 2 * params * ops_gen

    expvals = jnp.cos(par_ops_gates @ samples_gates.T)

    mean_expvals = jnp.mean(expvals, axis=-1)
    std_error = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)

    return mean_expvals, std_error


@jax.jit(static_argnames=["n_samples"])
def _mmd_mc_core(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ground_truth: jnp.ndarray,
    visible_ops: jnp.ndarray,
    all_ops: jnp.ndarray,
    key: Array,
    n_samples: int,
) -> float:
    """
    JIT core function to estimate MMD^2 loss.
    """
    n_qubits = generators.shape[1]
    m_train = ground_truth.shape[0]

    samples = jax.random.randint(key, shape=(n_samples, n_qubits), minval=0, maxval=2)

    ops_gen = (all_ops @ generators.T) % 2
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
    par_ops_gates = 2 * params * ops_gen

    tr_iqp_samples = jnp.cos(par_ops_gates @ samples_gates.T)

    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    correction = jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples

    tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)

    term_iqp = (tr_iqp**2 - correction) * n_samples / (n_samples - 1)
    term_cross = -2 * tr_iqp * tr_train
    term_train = (tr_train**2 * m_train - 1) / (m_train - 1)

    res = term_iqp + term_cross + term_train
    return jnp.mean(res)


def mmd_mc(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ground_truth: jnp.ndarray,
    sigma: float,
    n_ops: int,
    n_samples: int,
    key: Array,
    wires: Optional[List[int]] = None,
) -> float:
    """
    Return an unbiased estimate of the squared MMD loss of an IQP circuit.
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1 for unbiased estimation.")

    n_qubits = generators.shape[1]
    wires = wires if wires is not None else list(range(n_qubits))

    key, subkey = jax.random.split(key, 2)
    p_MMD = (1 - jnp.exp(-1 / (2 * sigma**2))) / 2
    visible_ops = jax.random.binomial(subkey, 1, p_MMD, shape=(n_ops, len(wires))).astype(jnp.float32)

    all_ops_list = []
    wire_idx = 0
    for q in range(n_qubits):
        if q in wires:
            all_ops_list.append(visible_ops[:, wire_idx])
            wire_idx += 1
        else:
            all_ops_list.append(jnp.zeros(n_ops))

    all_ops = jnp.array(all_ops_list, dtype=jnp.float32).T

    key, subkey_samples = jax.random.split(key, 2)

    loss = _mmd_mc_core(
        params=params,
        generators=generators,
        ground_truth=ground_truth,
        visible_ops=visible_ops,
        all_ops=all_ops,
        key=subkey_samples,
        n_samples=n_samples,
    )

    return float(loss)


def setup_training(
    init_params: jnp.ndarray,
    generators: jnp.ndarray,
    ground_truth: jnp.ndarray,
    sigma: float,
    n_ops: int,
    n_samples: int,
    lr: float = 0.01,
    wires: Optional[List[int]] = None,
):
    """
    Initialize optimizer state and define a JIT-compiled training step.
    """
    n_qubits = generators.shape[1]
    wires_arr = jnp.arange(n_qubits) if wires is None else jnp.array(wires)
    p_MMD = (1 - jnp.exp(-1 / (2 * sigma**2))) / 2

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)

    def loss_fn(params, key):
        key_ops, key_mc = jax.random.split(key, 2)
        all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float32)
        visible_ops = jax.random.binomial(key_ops, 1, p_MMD, shape=(n_ops, len(wires_arr))).astype(jnp.float32)
        all_ops = all_ops.at[:, wires_arr].set(visible_ops)

        return _mmd_mc_core(
            params=params,
            generators=generators,
            ground_truth=ground_truth,
            visible_ops=visible_ops,
            all_ops=all_ops,
            key=key_mc,
            n_samples=n_samples,
        )

    @jax.jit
    def train_step(params, opt_state, key):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    return opt_state, train_step


__all__ = [
    "expvals_contraction",
    "expvals_sampling",
    "expvals_mc",
    "_mmd_mc_core",
    "mmd_mc",
    "setup_training",
]
