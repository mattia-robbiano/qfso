from abc import ABC, abstractmethod
from functools import cache
from dataclasses import dataclass

from itertools import combinations

import numpy as np

from ..algebra import gf2_to_int, int_to_gf2, indices_to_gf2_matrix
from .utils import _build_product_distribution, _basis_permutation, _permute_distribution

class ProbabilityDistribution(ABC):

    n:int

    @abstractmethod
    def sample(self) -> int:
        pass

    @cache
    def vector(self) -> np.ndarray:
        return self._compute_vector()
    
    @cache
    def ks(self, hw_min, hw_max) -> np.ndarray:
        ks = []
        for h in range(hw_min, hw_max+1):
            for c in combinations(range(self.n), h):
                k = 0
                for bit in c:
                    k += 1<<bit
                ks.append(k)
        return np.array(ks)
    
    @cache
    def walsh_hadamard_spectrum(self, hw_min, hw_max) -> np.ndarray:
        return self._compute_walsh_hadamard_spectrum(hw_min, hw_max)

    @abstractmethod
    def _compute_vector(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _compute_walsh_hadamard_spectrum(self, hw_min, hw_max) -> np.ndarray:
        pass
