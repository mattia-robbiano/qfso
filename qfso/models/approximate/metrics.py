from abc import ABC
from functools import cache
import numpy as np

from .probability import ProbabilityDistribution

class Metric(ABC):
    pass

class MMD(Metric):
    
    def __init__(self, sigma:float, hw_min:int, hw_max:int) -> None:
        self.sigma = sigma
        self.hw_min = hw_min
        self.hw_max = hw_max

        p_sigma = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
        self.filter_value = lambda h: p_sigma**h*(1-p_sigma)**(1-h)
        self.multiplicity = lambda h, n: int(np.prod([i for i in range(n,n-h,-1)])/np.prod([i for i in range(1,h+1)]))

    @cache
    def filter(self, n:int):

        filter_generator = (
            [self.filter_value(h)]*self.multiplicity(h, n) for h in range(self.hw_min, self.hw_max+1)
        )
        return np.array(sum(filter_generator, start=[]))

    def __call__(self, p:ProbabilityDistribution, q:ProbabilityDistribution) -> float:

        p_hat = p.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        q_hat = q.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        return np.sum(self.filter(p.n)*(p_hat-q_hat)**2)
    
    def scalar_product(self, p:ProbabilityDistribution, q:ProbabilityDistribution):
        p_hat = p.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        q_hat = q.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        return np.sum(self.filter(p.n)*p_hat*q_hat)
    