from abc import ABC
from functools import cache
from math import comb
import numpy as np
import jax.numpy as jnp

from .probability import ProbabilityDistribution, FactorizedDistribution


class Metric(ABC):
    pass


class MMD(Metric):

    def __init__(self, sigma: float, hw_min: int, hw_max: int) -> None:
        self.sigma = sigma
        self.hw_min = hw_min
        self.hw_max = hw_max

        p_sigma = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
        self.filter_value = lambda h: p_sigma**h * (1 - p_sigma) ** (1 - h)
        # exact integer binomial coefficient - avoids overflow/precision
        # loss from the previous product-of-ranges formula once n~400
        self.multiplicity = lambda h, n: comb(n, h)

    @cache
    def filter(self, n: int):
        filter_generator = (
            [self.filter_value(h)] * self.multiplicity(h, n) for h in range(self.hw_min, self.hw_max + 1)
        )
        return jnp.asarray(sum(filter_generator, start=[]))

    def __call__(self, p: ProbabilityDistribution, q: ProbabilityDistribution) -> float:
        p_hat = p.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        q_hat = q.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        return jnp.sum(self.filter(p.n) * (p_hat - q_hat) ** 2)

    def scalar_product(self, p: ProbabilityDistribution, q: ProbabilityDistribution):
        p_hat = p.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        q_hat = q.walsh_hadamard_spectrum(self.hw_min, self.hw_max)
        return jnp.sum(self.filter(p.n) * p_hat * q_hat)

class SubsampledMMD(Metric):
    def __init__(self, n: int, sigma: float, hw_min: int, hw_max: int, fraction: float = 0.1) -> None:
        self.n = n
        self.sigma = sigma
        self.fraction = fraction
        
        from math import comb
        p_sigma = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
        filter_value = lambda h: p_sigma**h * (1 - p_sigma) ** (n - h)
        multiplicity = lambda h: comb(n, h)
        
        filter_list = list(filter_value(h) * multiplicity(h) for h in range(hw_min, hw_max + 1))
        self.all_weights = np.array(filter_list)
        dummy = FactorizedDistribution([1 << i for i in range(n)], [0.5] * n)
        self.all_ks = np.array(list(dummy.ks(hw_min, hw_max)))
        self.n_samples = int(len(self.all_ks) * fraction)
        
    def __call__(self, p: ProbabilityDistribution, q: ProbabilityDistribution) -> float:
        p_hat = p.walsh_hadamard_spectrum(ks=self.all_ks)
        q_hat = q.walsh_hadamard_spectrum(ks=self.all_ks)
        return jnp.sum(jnp.asarray(self.all_weights) * (p_hat - q_hat) ** 2)