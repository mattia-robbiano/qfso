import numpy as np

from qfso.distributions import uniform_like, discretized_normal_probability, plot_distributions
from qfso.models.approximate import match_first_order, match_mmd_optimal, mmd_squared, hamming_weight
from qfso.models.approximate.factorized import _sorted_mmd_contributions, match_mmd_multiple_learning, sample_mixture

n = 8
p = discretized_normal_probability((-2, 7), 2**n)
sigma = 0.1 * n
iterations=5

q1 = np.asarray(match_first_order(p))
mmd_q1 = mmd_squared(p, q1, sigma)

q2 = match_mmd_optimal(p, sigma=sigma, hw_min=1, hw_max=4)
mmd_q2 = mmd_squared(p, q2, sigma)

q3 = match_mmd_optimal(p, sigma=sigma, hw_min=1, hw_max=4, optimize=True)
mmd_q3 = mmd_squared(p, q3, sigma)


q4 = match_mmd_multiple_learning(p, sigma=sigma, hw_min=1, hw_max=4, optimize=True, number_of_distributions=iterations)

# define geometrically decreasing series of weights of len iterations summing to 1
weights = np.exp(-np.arange(iterations))
weights /= weights.sum()

# sampler for selecting from the mixture of distributions 
samplers = [
    lambda size, dist=q: np.random.choice(len(dist), size=size, p=dist)
    for q in q4
]
samples_mixture = sample_mixture(10_000, weights, samplers)
counts = np.bincount(samples_mixture.astype(int), minlength=2**n)
q4_empirical = counts / counts.sum()

mmd_q4 = mmd_squared(p, q4_empirical, sigma)


print(f"MMD²(p, first-order) = {mmd_q1:.6f}")
print(f"MMD²(p, mmd-optimal) = {mmd_q2:.6f}")
print(f"MMD²(p, mmd-learned) = {mmd_q3:.6f}")
print(f"MMD²(p, mixture) = {mmd_q4:.6f}")

#_____plotting_____

plot_distributions(
    [p, q1, q2, q3, q4_empirical],
    labels=["target distribution", f"match first-order (mmd={mmd_q1:.4e})", f"mmd-optimal (mmd={mmd_q2:.4e})", f"mmd-learned (mmd={mmd_q3:.4e})", f"mixture (mmd={mmd_q4:.4e})"],
    title=f"MMD-optimal factorized approximation {n}-qubit",
)