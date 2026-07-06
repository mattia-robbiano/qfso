import numpy as np

from .base import ProbabilityDistribution
from ..algebra import indices_to_gf2_matrix, int_to_gf2, gf2_to_int
from .utils import _basis_permutation, _build_product_distribution, _permute_distribution

class FactorizedDistribution(ProbabilityDistribution):

    def __init__(self, independent_parities: list[int], probabilities: list[float]):

        # Sanity check
        assert len(probabilities) == len(independent_parities), f"List of independent parities and probability must have the same length. Given ({len(probabilities)} and {len(independent_parities)})"
        self.n = len(independent_parities)

        self._independent_parities = independent_parities
        self.probabilities = probabilities

        matrix = indices_to_gf2_matrix(self.independent_parities, self.n)
        assert np.linalg.matrix_rank(matrix) == self.n, "Parities must be indipendent"
        matrix_forward = np.linalg.inv(matrix)
        matrix_backward = np.linalg.inv(matrix.T)

        self.forward_map = lambda parities: gf2_to_int(matrix_forward @ int_to_gf2(parities, self.n))
        self.backward_map = lambda sample: gf2_to_int(matrix_backward @ int_to_gf2(sample, self.n))

    @property
    def independent_parities(self,):
        return self._independent_parities

    @property
    def probabilities(self,):
        return self._probabilities
    
    @probabilities.setter
    def probabilities(self, ps:list[float]):

        assert len(ps) == self.n, "Probabilities must match the number of generators"
        for p in ps:
            assert p >= 0 and p <= 1, f"All probabilities must be in [0,1]. Given {p}"
        
        self._probabilities = ps
        self.vector.cache_clear()
        self.walsh_hadamard_spectrum.cache_clear()

    def sample(self) -> int:
        rs = np.random.random(size=(self.n,))

        sample = 0
        for i, (r,p) in enumerate(zip(rs, self.probabilities)):
            if r > p: sample += 1 << i

        return self.forward_map(sample)
    
    def _compute_vector(self) -> np.ndarray:
        vec = _build_product_distribution(np.array(self.probabilities))
        perm = _basis_permutation(self.independent_parities, self.n)
        return _permute_distribution(vec, perm)
    
    def _compute_walsh_hadamard_spectrum(self, hw_min, hw_max) -> np.ndarray:
        
        spectrum = []        
        single_site = [2*p-1 for p in self.probabilities]
            
        for k in self.ks(hw_min, hw_max):
            decomposition = self.backward_map(k)
            spectrum.append(
                np.prod([p_hat if (decomposition >> bit_position) & 1 else 1 for bit_position, p_hat in enumerate(single_site)])
            )

        return np.array(spectrum)