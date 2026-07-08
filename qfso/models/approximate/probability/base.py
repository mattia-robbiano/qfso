from abc import ABC, abstractmethod
from functools import cache
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import jax.numpy as jnp  # Aggiunto

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
    
    # Questo cache va bene perché hw_min e hw_max sono interi (hashable)
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
    
    def walsh_hadamard_spectrum(self, hw_min: int = None, hw_max: int = None, ks: np.ndarray = None) -> jnp.ndarray:
        """
        Interfaccia pubblica. Se ks non è fornito, lo genera usando hw_min e hw_max.
        Nota: rimosso @cache perché gli array NumPy non sono hashable.
        """
        if ks is None:
            if hw_min is None or hw_max is None:
                raise ValueError("Devi fornire 'ks' oppure sia 'hw_min' che 'hw_max'.")
            ks = self.ks(hw_min, hw_max)
        
        return self._compute_walsh_hadamard_spectrum(ks)
    
    @abstractmethod
    def _compute_vector(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _compute_walsh_hadamard_spectrum(self, ks: np.ndarray) -> jnp.ndarray:
        """Il metodo interno ora si aspetta sempre e solo l'array ks."""
        pass