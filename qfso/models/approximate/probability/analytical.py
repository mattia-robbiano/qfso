import numpy as np
from scipy.stats import norm

from .base import ProbabilityDistribution

class FromVector(ProbabilityDistribution):

    def __init__(self, vector):
        self._vector = vector
        self._xs = np.array(list(range(len(vector))))
        self.n = int(np.log2(len(vector)))
    
    def _compute_vector(self,):
        return self._vector
    
    def  _compute_walsh_hadamard_spectrum(self, hw_min, hw_max) -> np.ndarray:
    
        ks = self.ks(hw_min, hw_max)

        #actual sparse WHT. Costs ~ O(len(probs)*n^{hw_max})
        int_H = np.bitwise_and.outer(ks, self._xs)
        H = np.bitwise_count(int_H) % 2
        return 1 - 2*H@self._vector
    
    def sample(self,):
        raise NotImplementedError
    
class DiscretizedGaussian(FromVector):

    def __init__(self, n:int, loc:float=0, scale:float=1, left_limit:float=-3, right_limit:float=3):
        
        bins = np.linspace(left_limit, right_limit, 2**n + 1)
        probabilities = np.diff(norm.cdf(bins, loc=loc, scale=scale))
        discretized = probabilities / probabilities.sum()
        return super().__init__(discretized)