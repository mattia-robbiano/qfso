import galois
import numpy as np

GF = galois.GF2

def int_to_gf2(i: int, n: int) -> galois.Array:
    """Convert an integer to its binary representation in GF(2)."""
    return GF([(i >> j) & 1 for j in range(n)])

def gf2_to_int(v: galois.Array) -> int:
    """Convert a GF(2) array back to an integer."""
    return sum(int(v[j]) << j for j in range(len(v)))

def indices_to_gf2_matrix(indices: list[int], n: int) -> galois.Array:
    """Create a GF(2) matrix from a list of integer indices."""
    return GF([[(idx >> j) & 1 for j in range(n)] for idx in indices])

def biggest_independent_set(contributions: list[tuple], n: int) -> list[int]:
    """Find the largest independent set of generators based on matrix rank."""
    selected = []
    for idx, *_ in contributions:
        candidate = selected + [idx]
        matrix = indices_to_gf2_matrix(candidate, n)
        if np.linalg.matrix_rank(matrix) == len(candidate):
            selected.append(idx)
        if len(selected) == n:
            break
    return selected