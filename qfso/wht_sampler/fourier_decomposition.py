from dataclasses import dataclass
import numpy as np

from .basis_functions import BasisFunction, WalshHadamardBasisFunction


@dataclass
class FourierDecomposition:
    """
    Base class representing a distribution via its Fourier expansion.
    
    Attributes:
        basis_functions (list[BasisFunction]): The set of basis functions used.
        coefficients (list[complex]): The corresponding Fourier coefficients.
    """
    basis_functions: list['BasisFunction']
    coefficients: list[complex]

    def reset(self):
        """Resets all Fourier coefficients to zero."""
        self.coefficients = len(self.basis_functions) * [0]

    def from_samples(self, samples: list[int]):
        """
        Empirically estimates the Fourier coefficients from a set of target samples.
        
        Args:
            samples (list[int]): Integer representations of the sampled bitstrings.
        """
        for i, chi in enumerate(self.basis_functions):
            # Coefficient m_S = E_{x~p}[chi_S(x)]
            self.coefficients[i] = sum(map(chi, samples)) / len(samples)

class WalshHadamardDecomposition(FourierDecomposition):
    """
    Truncated Walsh-Hadamard expansion of a probability distribution over boolean hypercube.
    
    Provides methods to evaluate marginal probabilities of prefixes and to sample 
    using the Bremner-Montanaro-Shepherd (BMS) algorithm.
    """

    def __init__(self, identifiers: list[int], n: int):
        """
        Args:
            identifiers (list[int]): List of bitmasks defining the support (truncation) 
                                     of the Walsh-Hadamard transform.
            n (int): Total number of bits (qubits).
        """
        self.n = n
        super().__init__([WalshHadamardBasisFunction(i) for i in identifiers], len(identifiers) * [0])

    def marginal(self, prefix: int, bitlength: int = None) -> float:
        """
        Computes the marginal probability of a bitstring prefix.
        
        Args:
            prefix (int): The integer representation of the prefix bitstring.
            bitlength (int, optional): The length of the prefix. Inferred from prefix 
                                       if not provided, but explicitly needed for zero prefixes.
                                       
        Returns:
            float: The marginal probability mass of the given prefix.
        """
        bitlength = prefix.bit_length() if bitlength is None else bitlength
        prefix_end = 1 << max(bitlength, 1)

        m = 0.0
        for bf, c in zip(self.basis_functions, self.coefficients):
            # Contribute to marginal only if the basis function acts within the prefix
            if bf.identifier < prefix_end: 
                m += c * bf(prefix)

        return m / prefix_end

    def sample(self) -> int:
        """
        Samples a bitstring using the sequential BMS algorithm.
        
        This method iteratively evaluates the marginals for the next bit being 0 or 1.
        If the truncated representation produces a negative marginal mass for a branch,
        it assigns 0 probability to that branch and renormalizes the other, ensuring stability.
        
        Returns:
            int: The sampled bitstring as an integer.
        """
        rs = np.random.rand(self.n)
        s = 0

        for i, r in enumerate(rs):
            # Compute marginals for appending 0 (A) and appending 1 (B)
            A = self.marginal(s, bitlength=i+1)
            B = self.marginal(s + (1 << i), bitlength=i+1)

            # BMS Renormalization step: handling negative prefix masses
            if A < 0:
                A = 0
                B = 1.0 if B <= 0 else B / (B + A) # Safety check if both are <= 0
            if B < 0:
                B = 0
                A = 1.0 if A <= 0 else A / (A + B)

            # Assign next bit based on renormalized probabilities
            if r < (B / (A + B) if (A + B) > 0 else 0):
                s = s + (1 << i)
                
        return s