from abc import ABC
import numpy as np
import jax.numpy as jnp

from copy import copy
from time import time

from .algebra import indices_to_gf2_matrix
from .probability import ProbabilityDistribution, FactorizedDistribution, LinCombApproximation
from .metrics import MMD, SubsampledMMD
from .optimizer import fit

class BaseFitter(ABC):
    """Interfaccia base per tutti gli algoritmi di fitting delle distribuzioni."""
    pass


class DiscreteGreedyFitter(BaseFitter):
    """
    Costruisce una FactorizedDistribution identificando iterativamente i generatori 
    indipendenti dai coefficienti più alti dello spettro di Walsh-Hadamard.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.independent_parities = []
        self.coefficients = []
        self.sorted_ks = []
        self.sorted_spectrum = []

    def is_independent(self, candidate, n) -> bool:
        set = self.independent_parities + [candidate]
        matrix = indices_to_gf2_matrix(set, n)
        return np.linalg.matrix_rank(matrix) == len(set)

    def sort_contributions(self, ks, spectrum, top_n: int = None):
        """Filtra e ordina solo i top_n coefficienti di magnitudine maggiore."""
        ks = np.asarray(ks)
        spectrum = np.asarray(spectrum)
        abs_spectrum = np.abs(spectrum)

        if top_n is not None and top_n < len(abs_spectrum):
            candidate_idx = np.argpartition(abs_spectrum, -top_n)[-top_n:]
        else:
            candidate_idx = np.arange(len(abs_spectrum))

        order = candidate_idx[np.argsort(abs_spectrum[candidate_idx])[::-1]]
        return list(ks[order]), list(spectrum[order])

    # OLD ORA ACCETTA KS ESPLICITI
    # def fit(self, target: ProbabilityDistribution, hw_min: int, hw_max: int, top_n: int = None) -> FactorizedDistribution:
    #     self.sorted_ks, self.sorted_spectrum = self.sort_contributions(
    #         target.ks(hw_min, hw_max),
    #         target.walsh_hadamard_spectrum(hw_min, hw_max),
    #         top_n=top_n,
    #     )
    def fit(self, target: ProbabilityDistribution, ks: np.ndarray, top_n: int = None) -> FactorizedDistribution:
        self.sorted_ks, self.sorted_spectrum = self.sort_contributions(
            ks,
            target.walsh_hadamard_spectrum(ks=ks),
            top_n=top_n,
        )

        count = 0
        for k, c in zip(self.sorted_ks, self.sorted_spectrum):
            if self.is_independent(k, target.n):
                self.independent_parities.append(k)
                self.coefficients.append(np.clip(c, -1, 1))
                count += 1

            if count == target.n:
                return FactorizedDistribution(self.independent_parities, [0.5 * (1 + c) for c in self.coefficients])

        raise ValueError(
            "Could not find enough independent parities among the top_n candidates; "
            "consider broadening the spectrum, hw range, or increasing top_n."
        )


class OptimizedGreedyFitter(DiscreteGreedyFitter):
    """
    per ottimizzare i generatori
    """
    # def fit(self, target: ProbabilityDistribution, mmd: MMD, maxiter: int = 10, lr: float = 0.1, top_n: int = None) -> FactorizedDistribution:
    #     # Initial greedy selection of the generators
    #     p = super().fit(target, mmd.hw_min, mmd.hw_max, top_n=top_n)

    #     target_hat = target.walsh_hadamard_spectrum(mmd.hw_min, mmd.hw_max)
    #     ks = p.ks(mmd.hw_min, mmd.hw_max)
    #     mask = p.decomposition_mask(ks) # ~ matrice di quali generatori compongono quali frequenze, per evitare operazioni bitwise non diff
    #     filt = mmd.filter(p.n)          # e' la binomiale per i pesi

    #     def loss(params):
    #         # jax compute gradients of only this
    #         single_site = 2 * params - 1 
    #         p_hat = jnp.prod(jnp.where(mask, single_site[None, :], 1.0), axis=1)
    #         return jnp.sum(filt * (target_hat - p_hat) ** 2)

    #     p.probabilities = fit(loss, p.probabilities, n_iters=maxiter, lr=lr)
    #     return p

    # NOTA: Prende mmd come SubsampledMMD
    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, maxiter: int = 10, lr: float = 0.1, top_n: int = None) -> FactorizedDistribution:
        
        p = super().fit(target, mmd.active_ks, top_n=top_n)

        # Usiamo solo i ks campionati dalla metrica
        target_hat = target.walsh_hadamard_spectrum(ks=mmd.active_ks)
        mask = p.decomposition_mask(mmd.active_ks) # ~ matrice di quali generatori compongono quali frequenze, per evitare operazioni bitwise non diff
        filt = mmd.active_weights # e la binomiale con i pesi

        def loss(params):
            single_site = 2 * params - 1 
            p_hat = jnp.prod(jnp.where(mask, single_site[None, :], 1.0), axis=1)
            return jnp.sum(filt * (target_hat - p_hat) ** 2)

        p.probabilities = fit(loss, p.probabilities, n_iters=maxiter, lr=lr)
        return p


class FixedBasisFitter(BaseFitter):
    """
    Alternativa a OptimizedGreedyFitter che evita la ricerca dei generatori.
    Fissa i generatori alla base computazionale standard e ottimizza le 
    probabilità tramite gradiente stocastico calcolato su un subset di ks.
    """
    def reset(self):
        return

    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, maxiter: int = 10, lr: float = 0.1) -> FactorizedDistribution:
        # Estraiamo i coefficienti per hw=1 (sicuro anche con TruncatedArraySpectrum)
        probs = list(jnp.clip(0.5 * (1 + target.walsh_hadamard_spectrum(hw_min=1, hw_max=1)), 0, 1))
        p = FactorizedDistribution([1 << i for i in range(target.n)], probs)

        # Usiamo SOLO il subset stocastico pre-calcolato dalla metrica
        target_hat = target.walsh_hadamard_spectrum(ks=mmd.active_ks)
        ks = mmd.active_ks
        mask = p.decomposition_mask(ks)
        filt = mmd.active_weights

        def loss(params):
            single_site = 2 * params - 1
            p_hat = jnp.prod(jnp.where(mask, single_site[None, :], 1.0), axis=1)
            return jnp.sum(filt * (target_hat - p_hat) ** 2)

        p.probabilities = fit(loss, p.probabilities, n_iters=maxiter, lr=lr)
        return p

class IncrementalLinearCombBuilder(BaseFitter):
    """
    Metodo base per combinazioni lineari. Costruisce l'approssimazione in modo incrementale 
    aggiungendo nuove distribuzioni fattorizzate per fittare il residuo corrente. 
    """
    def __init__(self, n_probs: int):
        self.n_probs = n_probs
        super().__init__()

    # OLD
    # def fit(self, target: ProbabilityDistribution, mmd: MMD, top_n: int = None) -> LinCombApproximation:
    #     discrete_fitter = DiscreteGreedyFitter()
    #     factorized = discrete_fitter.fit(target, mmd.hw_min, mmd.hw_max, top_n=top_n)
    #     result = LinCombApproximation([factorized], [1.0])
    
    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, top_n: int = None) -> LinCombApproximation:
        discrete_fitter = DiscreteGreedyFitter()
        # Passiamo i ks attivi
        factorized = discrete_fitter.fit(target, mmd.active_ks, top_n=top_n)
        result = LinCombApproximation([factorized], [1.0])
        for i in range(2, self.n_probs + 1):
            current = LinCombApproximation([target, result], [1.0, -1.0])
            discrete_fitter.reset()
            # factorized = discrete_fitter.fit(current, mmd.hw_min, mmd.hw_max, top_n=top_n)
            factorized = discrete_fitter.fit(current, mmd.active_ks, top_n=top_n)
            result.append(factorized, 1 / i)
        return result


class SweepingLinearCombFitter(BaseFitter):
    """
    Ottimizzazione in stile sweep per linearcomb. 
    """
    def __init__(self, n_probs: int):
        self.n_probs = n_probs
        super().__init__()

    def sweep(self, target: ProbabilityDistribution, current: LinCombApproximation, mmd: MMD, maxiter: int, top_n: int = None):
        ps = copy(current.probabilities)
        ws = copy(current.weights)

        for i in range(self.n_probs):

            # se la mmd e quella stocastica resample
            if hasattr(mmd, 'resample'):
                mmd.resample()

            p = ps.pop(i)
            w = ws[i]
            reminder = LinCombApproximation([target] + ps, [1 / w] + (self.n_probs - 1) * [-1])
            self.single_prob_optimizer.reset()
            
            new_p = self.single_prob_optimizer.fit(reminder, mmd, maxiter, top_n=top_n) \
                if isinstance(self.single_prob_optimizer, OptimizedGreedyFitter) \
                else self.single_prob_optimizer.fit(reminder, mmd, maxiter)
            
            ps.insert(i, new_p)

        return LinCombApproximation(ps, ws)

    def fit(
        self,
        target: ProbabilityDistribution,
        mmd: MMD,
        sweeps: int = 10,
        it_per_sweep: int = 10,
        verbose: bool = False,
        fit_generators: bool = True,
        save_history: bool = False,
        top_n: int = None,
    ) -> LinCombApproximation:

        self.single_prob_optimizer = OptimizedGreedyFitter() if fit_generators else FixedBasisFitter()

        current = LinCombApproximation(
            self.n_probs * [
                FactorizedDistribution([1 << i for i in range(target.n)], target.n * [0.5])
            ],
            self.n_probs * [1 / self.n_probs],
        )

        self.history = []
        for i in range(sweeps):
            if verbose:
                t0_sweep = time()
                print(f"Sweep {i+1}..", flush=True, end="")
            if save_history:
                self.history.append(mmd(target, current))

            current = self.sweep(target, current, mmd, it_per_sweep, top_n=top_n)

            if verbose:
                print(f"Done [{(time()-t0_sweep):.2f}s] - MMD={self.history[-1]:.3e}", flush=True)

        if save_history:
            self.history.append(mmd(target, current))

        return current


class GlobalOptimizedLinearComb(IncrementalLinearCombBuilder):
    pass
#     """
#     Ottimizzazione globale 
#     """
#     def fit(self, target: ProbabilityDistribution, mmd: MMD, maxiter: int = 10, lr: float = 0.1, top_n: int = None, verbose: bool = False) -> LinCombApproximation:
#         # Costruzione greedy della struttura
#         approximation = super().fit(target, mmd, top_n=top_n)

#         dists = approximation.probabilities
#         sizes = [d.n for d in dists]
#         init_params = jnp.concatenate([jnp.asarray(d.probabilities) for d in dists])

#         masks = []
#         for d in dists:
#             ks = d.ks(mmd.hw_min, mmd.hw_max)
#             masks.append(d.decomposition_mask(ks))
            
#         target_hat = target.walsh_hadamard_spectrum(mmd.hw_min, mmd.hw_max)
#         filt = mmd.filter(target.n)
#         offsets = np.cumsum([0] + sizes)

#         def component_spectrum(params, i):
#             block = params[offsets[i]:offsets[i + 1]]
#             single_site = 2 * block - 1
#             return jnp.prod(jnp.where(masks[i], single_site[None, :], 1.0), axis=1)

#         weights = jnp.asarray(approximation.weights)

#         def loss(params):
#             p_hat = sum(w * component_spectrum(params, i) for i, w in enumerate(weights))
#             return jnp.sum(filt * (target_hat - p_hat) ** 2)

#         fitted = fit(loss, init_params, n_iters=maxiter, lr=lr)

#         for i, d in enumerate(dists):
#             d.probabilities = fitted[offsets[i]:offsets[i + 1]]

#         return approximation
