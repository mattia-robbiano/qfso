from dataclasses import dataclass
import numpy as np

from .base import ProbabilityDistribution


class LinCombApproximation(ProbabilityDistribution):

    def __init__(
            self,
            probabilities: list[ProbabilityDistribution], 
            weights: list[float], 
            is_valid_probability: bool = True
    ):
        # Sanity check
        assert len(probabilities) == len(weights), f"Linear Combination must contain the same amount of weights and probability distributions. Given ({len(probabilities)} and {len(weights)})"
        self.n = probabilities[0].n
        
        self.is_valid_probability = is_valid_probability
        self.probabilities = probabilities
        self.weights = weights
        
    @property
    def probabilities(self,):
        return self._probabilities
    
    @probabilities.setter
    def probabilities(self, ps):
        for p in ps:
            assert p.n == self.n, f"Probability distributions must all be defined on the same amount of bits. Given {self.n} and {p.n}" 
        
        self._probabilities = ps
        self.walsh_hadamard_spectrum.cache_clear()
        self.vector.cache_clear()
    
    @property
    def weights(self,):
        return self._weights
    
    @weights.setter
    def weights(self, ws):
        self.is_valid_probability = self.is_valid_probability and bool(np.isclose(sum(ws), 1.0, atol=1e-10))
        for w in ws:
            self.is_valid_probability = self.is_valid_probability and (w >= 0 and w <= 1)

        self._weights = ws 
        self.walsh_hadamard_spectrum.cache_clear()
        self.vector.cache_clear()

    def append(self, probability:ProbabilityDistribution, weight:float, renormalize_weights:bool = True):
        
        self.probabilities = self.probabilities + [probability]
        if renormalize_weights:
            self.weights = [(1-weight)*old_w for old_w in self.weights] + [weight]
        else:
            self.weights = self.weights + [weight]

    def sample(self) -> int:
        
        assert self.is_valid_probability, "Can only sample from a valid probability distribution"
        r = np.random.random()
        cdf = 0
        for w, p in zip(self.weights, self.probabilities):
            cdf += w
            if r < cdf: return p.sample()

        return self.probabilities[-1].sample()
    
    def _compute_vector(self) -> np.ndarray:

        vec = self.weights[0]*self.probabilities[0].vector()
        for w, p in zip(self.weights[1:], self.probabilities[1:]):
            vec += w*p.vector()
        return vec
    
    def _compute_walsh_hadamard_spectrum(self, hw_min, hw_max):

        specturm = self.weights[0]*self.probabilities[0].walsh_hadamard_spectrum(hw_min, hw_max)
        for w, p in zip(self.weights[1:], self.probabilities[1:]):
            specturm += w*p.walsh_hadamard_spectrum(hw_min, hw_max)
        return specturm