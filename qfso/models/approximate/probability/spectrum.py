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


# ATTENZIONE!! HO MESSO UNA PATCH TERRIBILE PERCHE IQPOPT DA COME EXPVALS UN DICT
# CAMBIA SOLO LA RIGA 54 CON raw_array["expvals"] - SIGNORE PERDONAMI
class TruncatedArraySpectrum(ProbabilityDistribution):
    def __init__(self, n: int, pkl_path: str, hw_min: int, hw_max: int):
        self.n = n
        
        with open(pkl_path, 'rb') as f:
            raw_array = pickle.load(f)
        raw_array=raw_array["expvals"]
        # Generiamo l'ordine canonico dei k per mappare l'array ai k corretti
        dummy = FactorizedDistribution([1 << i for i in range(n)], [0.5] * n)
        all_ks = list(dummy.ks(hw_min, hw_max))
        
        if len(raw_array) != len(all_ks):
            raise ValueError(
                f"Dimension mismatch: L'array caricato ha {len(raw_array)} elementi, "
                f"ma ci si aspetta {len(all_ks)} coefficienti per n={n}, hw=[{hw_min}, {hw_max}]."
            )
            
        # Dizionario di lookup per gestire in modo robusto il subsampling
        self._k_to_val = {int(k): float(v) for k, v in zip(all_ks, raw_array)}

    # 1. Implementa il metodo astratto per lo spettro richiesto da Base
    def _compute_walsh_hadamard_spectrum(self, ks: np.ndarray) -> jnp.ndarray:
        values = np.array([self._k_to_val.get(int(k), 0.0) for k in ks])
        return jnp.asarray(values)

    # 2. Blocca esplicitamente la generazione del vettore (impossibile per n=400)
    def _compute_vector(self) -> np.ndarray:
        raise NotImplementedError(
            "TruncatedArraySpectrum non supporta la generazione del vettore denso 2^n. "
            "Usa solo accessi sparsi tramite walsh_hadamard_spectrum()."
        )

    # 3. Blocca esplicitamente il sample (abbiamo solo uno spettro parziale)
    def sample(self) -> int:
        raise NotImplementedError(
            "TruncatedArraySpectrum non supporta il campionamento, "
            "poiché rappresenta solo uno spettro troncato."
        )