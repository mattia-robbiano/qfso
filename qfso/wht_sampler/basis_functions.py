class BasisFunction:
    """
    Base class for a basis function in a Fourier-like decomposition.
    
    Attributes:
        identifier (int): An integer representing the specific basis function 
                          (e.g., bitmask for Walsh-Hadamard).
    """

    def __init__(self, identifier: int):
        self.identifier = identifier

    def __call__(self, x: int) -> float:
        """
        Evaluates the basis function at a given point x.
        """
        raise NotImplementedError()

class WalshHadamardBasisFunction(BasisFunction):
    """
    Walsh-Hadamard basis function (also known as parity function).
    
    Computes chi_S(x) = (-1)^(S * x), where S is the subset of indices 
    encoded by the 'identifier' bitmask, and x is the input bitstring.
    In the context of IQP circuits, this corresponds to the expectation 
    value of a specific Pauli-Z string.
    """

    def __call__(self, x: int) -> int:
        """
        Evaluates the Walsh-Hadamard function.
        
        Args:
            x (int): The integer representation of the bitstring to evaluate.
            
        Returns:
            int: +1 if the bitwise dot product has an even number of 1s, -1 if odd.
        """
        prod = self.identifier & x
        # bit_count() returns the number of 1s (Hamming weight). 
        # Bitwise AND with 1 checks parity.
        return -1 if prod.bit_count() & 1 else 1