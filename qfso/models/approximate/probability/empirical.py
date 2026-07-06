import numpy as np

from .base import ProbabilityDistribution

class EmpiricalDistribution(ProbabilityDistribution):

    def __init__(self, n:int, samples:np.ndarray):
        self.n = n
        self.samples = samples

        vals, counts = np.unique(self.samples, return_counts=True)
        self.unique_values = vals
        self.normalized_counts = counts.astype(float) / len(self.samples)
    
    def  _compute_walsh_hadamard_spectrum(self, hw_min, hw_max) -> np.ndarray:
    
        ks = self.ks(hw_min, hw_max)

        #actual sparse WHT. Costs ~ O(len(samples)*n^{hw_max})
        int_H = np.bitwise_and.outer(ks,self.unique_values)
        H = np.bitwise_count(int_H) % 2
        return 1 - 2*H@self.normalized_counts
    
    def _compute_vector(self) -> np.ndarray:

        vec = np.zeros(2**self.n)
        for val, norm_counts in zip(self.unique_values, self.normalized_counts):
            vec[val] = norm_counts 
        return vec
