import numpy as np
import jax.numpy as jnp

from .base import ProbabilityDistribution
from ..algebra import indices_to_gf2_matrix, int_to_gf2, gf2_to_int
from .utils import _basis_permutation, _build_product_distribution, _permute_distribution


class FactorizedDistribution(ProbabilityDistribution):

    def __init__(self, independent_parities: list, probabilities):

        assert len(probabilities) == len(independent_parities), (
            f"List of independent parities and probability must have the same length. "
            f"Given ({len(probabilities)} and {len(independent_parities)})"
        )
        self.n = len(independent_parities)
        self._independent_parities = independent_parities
        self.probabilities = probabilities

        matrix = indices_to_gf2_matrix(self.independent_parities, self.n)
        assert np.linalg.matrix_rank(matrix) == self.n, "Parities must be independent"
        matrix_forward = np.linalg.inv(matrix)
        matrix_backward = np.linalg.inv(matrix.T)

        self.forward_map = lambda parities: gf2_to_int(matrix_forward @ int_to_gf2(parities, self.n))
        self.backward_map = lambda sample: gf2_to_int(matrix_backward @ int_to_gf2(sample, self.n))

    @property
    def independent_parities(self):
        return self._independent_parities

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, ps):
        assert len(ps) == self.n, "Probabilities must match the number of generators"
        for p in ps:
            assert p >= 0 and p <= 1, f"All probabilities must be in [0,1]. Given {p}"

        # Stored as a jnp array so this can flow through jax.grad / jax.jit
        self._probabilities = jnp.asarray(ps, dtype=jnp.float32)
        self.vector.cache_clear()
        # RIMOSSO: self.walsh_hadamard_spectrum.cache_clear()

    def sample(self) -> int:
        rs = np.random.random(size=(self.n,))
        probs = np.asarray(self.probabilities)

        sample = 0
        for i, (r, p) in enumerate(zip(rs, probs)):
            if r > p:
                sample += 1 << i

        return self.forward_map(sample)

    def decomposition_mask(self, ks: np.ndarray) -> jnp.ndarray:
        """
        Static (non-differentiable) bit-mask of shape (len(ks), n): entry
        [i, bit] is True iff generator `bit` participates in k_i's
        decomposition over this basis. Computed once with plain numpy -
        cheap even for n=400 since it only scales with len(ks), not 2**n.
        """
        decompositions = np.array([int(self.backward_map(int(k))) for k in ks])
        bits = np.arange(self.n)
        mask = ((decompositions[:, None] >> bits[None, :]) & 1).astype(bool)
        return jnp.asarray(mask)

    def _compute_vector(self) -> np.ndarray:
        vec = _build_product_distribution(np.asarray(self.probabilities))
        perm = _basis_permutation(self.independent_parities, self.n)
        return _permute_distribution(vec, perm)

    # AGGIORNATO: Ora accetta direttamente l'array 'ks'
    def _compute_walsh_hadamard_spectrum(self, ks: np.ndarray) -> jnp.ndarray:
        mask = self.decomposition_mask(ks)
        single_site = 2 * self.probabilities - 1
        terms = jnp.where(mask, single_site[None, :], 1.0)
        return jnp.prod(terms, axis=1)