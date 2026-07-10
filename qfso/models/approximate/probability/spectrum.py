import numpy as np
import jax.numpy as jnp

from .base import ProbabilityDistribution
import pickle
import numpy as np
import jax.numpy as jnp
from .base import ProbabilityDistribution
from .factorized import FactorizedDistribution


class FromSpectrum(ProbabilityDistribution):
    """
    A distribution defined directly by a (sparse) set of Walsh-Hadamard
    coefficients. Use this to 'spoof' a target distribution when you only
    have precomputed Fourier coefficients and n is too large (e.g. 400)
    to ever materialize a length-2^n probability vector.

    `spectrum` only needs to contain the coefficients you actually care
    about (i.e. the ones you'll query via hw_min/hw_max downstream, for
    example when computing an MMD against a FactorizedDistribution).
    Any k not present defaults to 0.
    """

    def __init__(self, n: int, spectrum: dict):
        self.n = n
        self._spectrum = {int(k): float(v) for k, v in spectrum.items()}

    def _compute_walsh_hadamard_spectrum(self, hw_min, hw_max) -> jnp.ndarray:
        ks = self.ks(hw_min, hw_max)
        values = np.array([self._spectrum.get(int(k), 0.0) for k in ks])
        return jnp.asarray(values)

    def _compute_vector(self):
        # Intentionally not supported: for n=400, 2**n is not representable.
        raise NotImplementedError(
            "FromSpectrum has no dense vector representation for large n. "
            "Only sparse Walsh-Hadamard access via walsh_hadamard_spectrum() "
            "is supported."
        )

    def sample(self):
        raise NotImplementedError("FromSpectrum does not support sampling.")


        raise NotImplementedError(
            "TruncatedArraySpectrum non supporta il campionamento, "
            "poiché rappresenta solo uno spettro troncato."
        )


class TruncatedArraySpectrum(ProbabilityDistribution):

    def __init__(self, n: int, pkl_path: str, hw_min: int, hw_max: int, file_hw_min: int = 1, file_hw_max: int = 2):
        self.n = n
        with open(pkl_path, 'rb') as f:
            raw_array = pickle.load(f)["expvals"]

        # 1. Genera i k per mappare la struttura fisica dell'array nel file
        dummy = FactorizedDistribution([1 << i for i in range(n)], [0.5] * n)
        file_ks = list(dummy.ks(file_hw_min, file_hw_max))
        
        if len(raw_array) != len(file_ks):
            raise ValueError(
                f"Dimension mismatch: L'array caricato ha {len(raw_array)} elementi, "
                f"ma file_hw=[{file_hw_min}, {file_hw_max}] si aspetta {len(file_ks)} coefficienti per n={n}."
            )
            
        # 2. Crea il dizionario completo mappando il file
        full_k_to_val = {int(k): float(v) for k, v in zip(file_ks, raw_array)}

        # 3. Filtra e conserva solo il range richiesto per l'addestramento (hw_min, hw_max)
        target_ks = set(dummy.ks(hw_min, hw_max))
        self._k_to_val = {k: v for k, v in full_k_to_val.items() if k in target_ks}

    def _compute_walsh_hadamard_spectrum(self, ks: np.ndarray) -> jnp.ndarray:
        values = np.array([self._k_to_val.get(int(k), 0.0) for k in ks])
        return jnp.asarray(values)

    def _compute_vector(self) -> np.ndarray:
        raise NotImplementedError(
            "TruncatedArraySpectrum non supporta la generazione del vettore denso 2^n. "
            "Usa solo accessi sparsi tramite walsh_hadamard_spectrum()."
        )

    def sample(self) -> int:
        raise NotImplementedError(
            "TruncatedArraySpectrum non supporta il campionamento, "
            "poiché rappresenta solo uno spettro troncato."
        )