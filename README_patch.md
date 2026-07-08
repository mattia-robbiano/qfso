# Patch notes

Drop these files in over the existing package (same relative paths). Files
not listed here (`base.py`, `empirical.py`, `analytical.py`, `lin_comb.py`,
`utils.py`, `algebra.py`) are untouched — they're only used at small n for
tests and don't sit on the n=400 path.

## 1. Audit: is the full 2^n spectrum computed anywhere?

No. Every `_compute_walsh_hadamard_spectrum` (Factorized, Empirical,
FromVector, LinComb) is already sparse — it only evaluates the `ks` you
ask for via `hw_min`/`hw_max`, never the full 2^n set. The only things that
scale as 2^n are the `_compute_vector()` methods (`FactorizedDistribution`,
`EmpiricalDistribution`, `FromVector`), and those are never called by the
MMD/heuristics training path — only `sample()` and
`walsh_hadamard_spectrum()` are used there. So the training loop is already
n-agnostic; nothing needed fixing here. `FromSpectrum._compute_vector`
below explicitly raises rather than silently trying to build one.

## 2. `probability/spectrum.py` — `FromSpectrum`

New distribution class for the "spoofed" target: pass `n` and a
`{k: coefficient}` dict of whatever Walsh-Hadamard coefficients you have.
No 2^n vector, no sampling — just sparse lookups, so `n=400` costs nothing
extra. Exported from `probability/__init__.py` and the top-level package.

```python
target = FromSpectrum(n=400, spectrum={1: 0.12, 2: -0.05, ...})
```

## 3. `probability/factorized.py` — JAX-ified

`FactorizedDistribution.probabilities` is now stored as a `jnp` array.
`_compute_walsh_hadamard_spectrum` precomputes a static boolean mask
(`decomposition_mask`, plain numpy, O(len(ks) * n) — cheap even at n=400)
once, then computes the spectrum as a pure `jnp` expression of
`probabilities`. This is what makes the fits below differentiable/jittable.
`_compute_vector` is untouched (still only viable for small n — nothing
calls it during training).

## 4. `metrics.py` — JAX-ified MMD

Same shape as before, just `jnp` instead of `np` so it composes with
`jax.grad`. Also switched the multiplicity formula to `math.comb` (exact
integer binomial coefficient) instead of a raw product-of-ranges, which
was liable to lose precision/overflow once `n` gets into the hundreds.

## 5. `optimizer.py` — replaces `scipy.optimize.minimize`

`gradient_descent_fit(loss_fn, init_params, n_iters=10, lr=0.1)`: a minimal
Adam optimizer, `loss_fn` must be a pure `params -> scalar` function. The
whole loop runs through `jax.lax.scan`, so it JIT-compiles end to end —
no more per-iteration Python/scipy overhead once you're doing this
thousands of times across generators.

## 6. `heuristics.py`

- `MaxCoefficientHeuristic.sort_contributions` takes a new `top_n`
  argument: uses `np.argpartition` to grab only the top-`top_n`
  largest-magnitude coefficients, and sorts just those, instead of sorting
  the entire spectrum. Pass `top_n=None` to fall back to the old
  full-sort behaviour. Threaded through `fit()` and the linear-comb
  heuristics.
- `OptimizedMaxCoeffHeuristic.fit` / `OptimizedOnlyCoeffs.fit` /
  `OptimizedMaxLinearCombHeuristic.fit` now build the loss as a pure `jnp`
  function (closing over precomputed masks/targets/filters) and call
  `gradient_descent_fit` instead of `scipy.optimize.minimize`.
- `OptimizedMaxLinearCombHeuristic.fit` fits all component distributions'
  probabilities jointly in one optimizer call (concatenated parameter
  vector, sliced per component) rather than component-by-component.

## Notes / things to pick when you actually run n=400

- `top_n` needs to be large enough that a rank-`n` independent set exists
  among the candidates — if `MaxCoefficientHeuristic.fit` raises
  "Could not find enough independent parities", raise `top_n`. With random
  coefficients you generally want `top_n` a small multiple of the number
  of hw=1..hw_max candidates, not just `n`.
- `gradient_descent_fit`'s `lr`/`n_iters` were left at small defaults
  matching "few iterations" — tune once you see real loss curves.
- Nothing here changes `hw_max` complexity: `ks()` still uses
  `itertools.combinations(range(n), h)`, so keep `hw_max` small (1 or 2)
  at n=400 or the candidate set itself explodes combinatorially — that's
  independent of the JAX/top_n work.
