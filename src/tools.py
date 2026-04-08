from itertools import combinations
import numpy as np
from scipy.stats import norm

from utils import WH_fixed_order_ids
from wht_sampler import WalshHadamardDecomposition, WalshHadamardBasisFunction

def discretized_normal_probability(interval: tuple[float, float], num_bins: int, loc: float = 0) -> np.ndarray:
    """
    Generates a discrete probability distribution from a continuous Gaussian.
    
    Args:
        interval (tuple): (min_val, max_val) specifying the integration range.
        num_bins (int): Number of discrete states.
        loc (float): Mean of the normal distribution.
        
    Returns:
        np.ndarray: Normalized 1D array of discrete probabilities.
    """
    min_val, max_val = interval
    bins = np.linspace(min_val, max_val, num_bins + 1)
    probabilities = []
    
    for i in range(num_bins):
        prob = norm.cdf(bins[i+1], loc=loc, scale=1) - norm.cdf(bins[i], loc=loc, scale=1)
        probabilities.append(prob)
        
    probability_vector = np.array(probabilities)
    probability_vector /= np.sum(probability_vector) #normalize
    return probability_vector

def exact_WH_coefficient(probability_vector: np.ndarray, identifier: int) -> float:
    """
    Computes the exact Walsh-Hadamard coefficient for a given distribution and identifier.
    
    Args:
        probability_vector (np.ndarray): The full exact probability distribution.
        identifier (int): Bitmask representing the Pauli/parity string.
        
    Returns:
        float: The exact Fourier coefficient.
    """
    basis_func = WalshHadamardBasisFunction(identifier)
    return sum([p * basis_func(i) for i, p in enumerate(probability_vector)])



def convergence_scaling(n: int, order: int | list[int], probability_vector: np.ndarray, 
                        min_samples: int = 100, max_samples: int = 10_000, 
                        num_points: int = 2, repetitions_per_point: int = 1):
    """
    Evaluates the convergence of empirical WHT coefficients to their exact values 
    as a function of the number of samples.
    
    Args:
        n (int): Number of bits.
        order (int | list[int]): Truncation orders to evaluate.
        probability_vector (np.ndarray): Exact probability array to sample from.
        min_samples (int): Starting sample count.
        max_samples (int): Final sample count.
        num_points (int): Number of grid points for sample sizes.
        repetitions_per_point (int): Number of statistical repetitions per sample size.
        
    Yields:
        np.ndarray: The absolute error vector between empirical mean and exact coefficients.
    """
    order_ids = WH_fixed_order_ids(n=n, order=order)
    true_coefficients = np.array([exact_WH_coefficient(probability_vector, i) for i in order_ids])

    # Utilizziamo dtype=int per assicurarci che numpy iteri su interi validi per il size
    sample_sizes = np.linspace(min_samples, max_samples, num_points, dtype=int)

    for num_samples in sample_sizes:
        coeffs_matrix = []
        for _ in range(repetitions_per_point):
            fd = WalshHadamardDecomposition(identifiers=order_ids, n=n)
            
            # Simulated dataset from the ground-truth distribution
            samples = np.random.choice(
                np.arange(probability_vector.shape[0]), 
                p=probability_vector, 
                size=num_samples
            )
            
            fd.from_samples(samples)
            coeffs_matrix.append(fd.coefficients)
        
        mean_estimate = np.mean(coeffs_matrix, axis=0)
        yield np.abs(mean_estimate - true_coefficients)