from itertools import combinations

def WH_fixed_order_ids(n: int, order: int | list[int]) -> list[int]:
    """
    Generates all Walsh-Hadamard identifiers (bitmasks) up to a specific Hamming weight.
    
    This is used to construct a *truncated* Walsh-Fourier representation, equivalent 
    to tracking only low-weight Pauli observables.
    
    Args:
        n (int): Total number of bits (or qubits).
        order (int | list[int]): The specific Hamming weight(s) of the bitstrings to include.
        
    Returns:
        list[int]: A list of integer bitmasks.
    """
    if isinstance(order, int):
        order = [order]
    
    ids = []
    for k in order:
        # Generate all subset combinations of size k
        for indices in combinations(range(n), k):
            ident = 0
            for i in indices:
                ident += (1 << i)
            ids.append(ident)
    return ids