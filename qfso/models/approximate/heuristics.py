from abc import ABC
import numpy as np
import jax
import jax.numpy as jnp

from copy import copy
from time import time

from .algebra import indices_to_gf2_matrix
from .probability import ProbabilityDistribution, FactorizedDistribution, LinCombApproximation, TruncatedArraySpectrum
from .metrics import MMD, SubsampledMMD
from .optimizer import fit_stochastic

class BaseFitter(ABC):
    pass


class DiscreteGreedyFitter(BaseFitter):
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
        ks = np.asarray(ks)
        spectrum = np.asarray(spectrum)
        abs_spectrum = np.abs(spectrum)

        if top_n is not None and top_n < len(abs_spectrum):
            candidate_idx = np.argpartition(abs_spectrum, -top_n)[-top_n:]
        else:
            candidate_idx = np.arange(len(abs_spectrum))

        order = candidate_idx[np.argsort(abs_spectrum[candidate_idx])[::-1]]
        return list(ks[order]), list(spectrum[order])

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
    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, maxiter: int = 10, lr: float = 0.1, top_n: int = None) -> FactorizedDistribution:
        
        p = super().fit(target, mmd.all_ks, top_n=top_n)

        target_hat = target.walsh_hadamard_spectrum(ks=mmd.all_ks)
        mask = p.decomposition_mask(mmd.all_ks) 
        filt = jnp.asarray(mmd.all_weights)

        def loss(params):
            single_site = 2 * params - 1 
            p_hat = jnp.prod(jnp.where(mask, single_site[None, :], 1.0), axis=1)
            return jnp.sum(filt * (target_hat - p_hat) ** 2)

        p.probabilities = fit(loss, p.probabilities, n_iters=maxiter, lr=lr)
        return p


class FixedBasisFitter(BaseFitter):
    def __init__(self):
        super().__init__()
        self._mask_cache = None
        self._mask_ks_id = None

    def reset(self):
        pass

    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, maxiter: int = 10, lr: float = 0.1) -> FactorizedDistribution:
        probs = list(jnp.clip(0.5 * (1 + target.walsh_hadamard_spectrum(hw_min=1, hw_max=1)), 0, 1))
        p = FactorizedDistribution([1 << i for i in range(target.n)], probs)

        all_ks = mmd.all_ks
        n_samples = mmd.n_samples
        filt_all = jnp.asarray(mmd.all_weights)
        
        if self._mask_ks_id != id(all_ks):
            self._mask_cache = p.decomposition_mask(all_ks)
            self._mask_ks_id = id(all_ks)
        mask_all = self._mask_cache

        target_hat_all = target.walsh_hadamard_spectrum(ks=all_ks)

        # Calculate sizes outside JIT using standard numpy/python
        filt_np = np.asarray(mmd.all_weights)
        norm = np.sum(filt_np)
        len_hw1 = target.n
        len_hw2 = len(all_ks) - target.n

        ideal_samples1 = int(n_samples * (filt_np[0] / norm))
        
        # CRITICAL: Prevent "replace=False"
        # ma penso che si possa pure togliere
        n_samples1 = min(ideal_samples1, len_hw1)
        n_samples2 = min(n_samples - n_samples1, len_hw2)

        def loss(params, subkey, target_full, mask_full, filt_full):
            key1, key2 = jax.random.split(subkey)
            
            # jax.random.choice on an integer 'x' samples from the range [0, x).
            # We shift the second block by len_hw1 to target the HW=2 section of the arrays.
            idx1 = jax.random.choice(key1, len_hw1, shape=(n_samples1,), replace=False)
            idx2 = jax.random.choice(key2, len_hw2, shape=(n_samples2,), replace=False) + len_hw1
            
            # Concatenate to perform the heavy matrix math in one single vectorized pass
            batch_idx = jnp.concatenate([idx1, idx2])
            batch_mask = mask_full[batch_idx]
            batch_target = target_full[batch_idx]

            single_site = 2 * params - 1
            p_hat = jnp.prod(jnp.where(batch_mask, single_site[None, :], 1.0), axis=1)
            
            # Split the results back apart using static slicing to compute independent means
            mean1 = jnp.mean((batch_target[:n_samples1] - p_hat[:n_samples1]) ** 2)
            mean2 = jnp.mean((batch_target[n_samples1:] - p_hat[n_samples1:]) ** 2)
            
            return filt_full[0] * mean1 + filt_full[1] * mean2

        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        # Passiamo i tensori esplicitamente al JIT
        p.probabilities, final_batch_loss = fit_stochastic(
            loss, 
            p.probabilities, 
            key, 
            target_hat_all, 
            mask_all, 
            filt_all, 
            n_iters=maxiter, 
            lr=lr
        )
        return p, final_batch_loss


class IncrementalLinearCombBuilder(BaseFitter):
    def __init__(self, n_probs: int):
        self.n_probs = n_probs
        super().__init__()
    
    def fit(self, target: ProbabilityDistribution, mmd: SubsampledMMD, top_n: int = None) -> LinCombApproximation:
        discrete_fitter = DiscreteGreedyFitter()
        factorized = discrete_fitter.fit(target, mmd.all_ks, top_n=top_n)
        result = LinCombApproximation([factorized], [1.0])
        for i in range(2, self.n_probs + 1):
            current = LinCombApproximation([target, result], [1.0, -1.0])
            discrete_fitter.reset()
            factorized = discrete_fitter.fit(current, mmd.all_ks, top_n=top_n)
            result.append(factorized, 1 / i)
        return result


class SweepingLinearCombFitter(BaseFitter):
    def __init__(self, n_probs: int):
        self.n_probs = n_probs
        super().__init__()

    def sweep(self, target: TruncatedArraySpectrum, current: LinCombApproximation, mmd: MMD, maxiter: int, top_n: int = None):
            ps = copy(current.probabilities)
            ws = copy(current.weights)
            sweep_loss = None # Variabile per salvare la loss

            for i in range(self.n_probs):
                p = ps.pop(i)
                w = ws[i]
                reminder = LinCombApproximation([target] + ps, [1 / w] + (self.n_probs - 1) * [-1])
                self.single_prob_optimizer.reset()
                
                result = self.single_prob_optimizer.fit(reminder, mmd, maxiter, top_n=top_n) \
                    if isinstance(self.single_prob_optimizer, OptimizedGreedyFitter) \
                    else self.single_prob_optimizer.fit(reminder, mmd, maxiter)
                
                print("*", flush=True, end=" ")

                # la loss dell ottimizzatore e relativa al resto, bisogna moltiplicare per
                # w^2 per ottenere la completa
                if isinstance(result, tuple):
                    new_p, comp_loss = result
                    sweep_loss = float(comp_loss) * (w ** 2)
                else:
                    new_p = result
                
                ps.insert(i, new_p)

            # Restituiamo anche la loss dello sweep
            return LinCombApproximation(ps, ws), sweep_loss

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
                self.n_probs * [FactorizedDistribution([1 << i for i in range(target.n)], target.n * [0.5])],
                self.n_probs * [1 / self.n_probs],
            )

            self.history = []
            for i in range(sweeps):
                if verbose:
                    t0_sweep = time()
                    print(f"Sweep {i+1}..", flush=True, end="")

                # Ora sweep restituisce la tupla
                current, last_loss = self.sweep(target, current, mmd, it_per_sweep, top_n=top_n)

                if save_history:
                    if last_loss is not None:
                        self.history.append(last_loss)
                    else:
                        # Fallback nel caso si usi un fitter vecchio (es. OptimizedGreedyFitter)
                        self.history.append(float(mmd(target, current)))

                if verbose:
                    loss_str = f"{self.history[-1]:.3e}" if save_history else "N/A"
                    print(f"Done [{(time()-t0_sweep):.2f}s] - MMD={loss_str}", flush=True)

            return current