from __future__ import annotations

import galois
import numpy as np

from qfso.distributions.transform.wh.utils import get_wh_coefficients_in_range

GF = galois.GF2


def int_to_gf2(i: int, n: int) -> galois.Array:
    return GF([(i >> j) & 1 for j in range(n)])


def gf2_to_int(v: galois.Array) -> int:
    return sum(int(v[j]) << j for j in range(len(v)))


def indices_to_gf2_matrix(indices: list[int], n: int) -> galois.Array:
    return GF([[(idx >> j) & 1 for j in range(n)] for idx in indices])


def hamming_weight(k: int) -> int:
    return k.bit_count()


def biggest_independent_set(contributions: list, n: int) -> list[int]:
    selected: list[int] = []
    for idx, *_ in contributions:
        candidate = selected + [idx]
        matrix = indices_to_gf2_matrix(candidate, n)
        if np.linalg.matrix_rank(matrix) == len(candidate):
            selected.append(idx)
        if len(selected) == n:
            break
    return selected


def sorted_mmd_contributions(
    distribution: np.ndarray, sigma: float, hw_min: int = 1, hw_max: int | None = None
) -> list[tuple[int, float, float]]:
    from .metrics import mmd_kernel_weight

    n = int(round(np.log2(distribution.size)))
    hw_max = n if hw_max is None else hw_max
    fourier_coeffs = get_wh_coefficients_in_range(distribution, n, hw_min, hw_max)
    result = [
        (k, mmd_kernel_weight(hamming_weight(k), n, sigma) * c**2, float(c))
        for k, c in fourier_coeffs.items()
    ]
    result.sort(key=lambda item: item[1], reverse=True)
    return result


__all__ = [
    "GF",
    "int_to_gf2",
    "gf2_to_int",
    "indices_to_gf2_matrix",
    "hamming_weight",
    "biggest_independent_set",
    "sorted_mmd_contributions",
]