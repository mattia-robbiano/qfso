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

        # Fast path: the "standard basis" (one single bit per generator, in
        # order) is exactly the identity matrix in GF(2) -> rank is trivially
        # n and its inverse is itself. This basis is what's used every time a
        # fresh FactorizedDistribution is built from scratch (the initial
        # `current` in SweepingLinearCombFitter.fit, and every single call of
        # FixedBasisFitter.fit), so skipping the O(n^3) GF(2) matrix
        # construction/rank-check/inversion here gives identical results for
        # a fraction of the cost, with no change to the general-basis path
        # used by DiscreteGreedyFitter/OptimizedGreedyFitter.
        self._is_standard_basis = independent_parities == [1 << i for i in range(self.n)]

        if self._is_standard_basis:
            self.forward_map = lambda parities: parities
            self.backward_map = lambda sample: sample
        else:
            matrix = indices_to_gf2_matrix(self.independent_parities, self.n)
            assert np.linalg.matrix_rank(matrix) == self.n, "Parities must be independent"
            matrix_forward = np.linalg.inv(matrix)
            matrix_backward = np.linalg.inv(matrix.T)

            self.forward_map = lambda parities: gf2_to_int(matrix_forward @ int_to_gf2(parities, self.n))
            self.backward_map = lambda sample: gf2_to_int(matrix_backward @ int_to_gf2(sample, self.n))

        # Caching per evitare ricalcoli costosi durante lo sweep
        self._cached_ks_id = None
        self._cached_mask = None

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
            assert 0 <= p <= 1, f"All probabilities must be in [0,1]. Given {p}"

        self._probabilities = jnp.asarray(ps, dtype=jnp.float32)
        self.vector.cache_clear()

    def sample(self) -> int:
        rs = np.random.random(size=(self.n,))
        probs = np.asarray(self.probabilities)

        sample = 0
        for i, (r, p) in enumerate(zip(rs, probs)):
            if r > p:
                sample += 1 << i

        return self.forward_map(sample)

    def decomposition_mask(self, ks: np.ndarray) -> jnp.ndarray:
        ks_id = id(ks)
        if self._cached_ks_id == ks_id and self._cached_mask is not None:
            return self._cached_mask

        if self._is_standard_basis:
            # backward_map is the identity here, so this skips a Python-level
            # loop of matrix-vector products (one per k) for no change in result.
            # NOTE: no dtype is forced here on purpose - k values can exceed
            # int64 for large n (bitmasks over n-1 bits), same as the original
            # per-element loop below, which relies on numpy's default dtype
            # inference (falling back to dtype=object for huge Python ints).
            decompositions = np.asarray(ks)
        else:
            decompositions = np.array([int(self.backward_map(int(k))) for k in ks])
        bits = np.arange(self.n)
        mask = ((decompositions[:, None] >> bits[None, :]) & 1).astype(bool)
        
        self._cached_mask = jnp.asarray(mask)
        self._cached_ks_id = ks_id
        return self._cached_mask

    def _compute_vector(self) -> np.ndarray:
        vec = _build_product_distribution(np.asarray(self.probabilities))
        perm = _basis_permutation(self.independent_parities, self.n)
        return _permute_distribution(vec, perm)

    def _compute_walsh_hadamard_spectrum(self, ks: np.ndarray) -> jnp.ndarray:
        mask = self.decomposition_mask(ks)
        single_site = 2 * self.probabilities - 1
        terms = jnp.where(mask, single_site[None, :], 1.0)
        return jnp.prod(terms, axis=1)