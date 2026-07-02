import numpy as np

from qfso.distributions import uniform_like, discretized_normal_probability, plot_distributions
from qfso.models.approximate import match_first_order,match_mmd_optimal,mmd_squared, hamming_weight
from qfso.models.approximate.factorized import _sorted_mmd_contributions

n = 8
p = discretized_normal_probability((-10, 9), 2**n)

sigma = 0.1 * n
q1 = np.asarray(match_first_order(p))
q2 = match_mmd_optimal(p, sigma=sigma, hw_min=1, hw_max=4)
q3 = match_mmd_optimal(p, sigma=sigma, hw_min=1, hw_max=4, optimize=True)

mmd_q1 = mmd_squared(p, q1, sigma)
mmd_q2 = mmd_squared(p, q2, sigma)
mmd_q3 = mmd_squared(p, q3, sigma)

print(f"MMD²(p, first-order) = {mmd_q1:.6f}")
print(f"MMD²(p, mmd-optimal) = {mmd_q2:.6f}")
print(f"MMD²(p, mmd-learned) = {mmd_q3:.6f}")

# plot_distributions(
#     [p, q1, q2, q3],
#     labels=["target distribution", f"match first-order (mmd={mmd_q1:.4e})", f"mmd-optimal (mmd={mmd_q2:.4e})", f"mmd-learned (mmd={mmd_q3:.4e})"],
#     title=f"MMD-optimal factorized approximation {n}-qubit",
# )

# contributions = _sorted_mmd_contributions(p, sigma=sigma, hw_min=0, hw_max=3)
# print("\nTop MMD contributions:")
# for k, contrib, c in contributions[:10]:
#     print(f"  k={k:0{n}b}  hw={hamming_weight(k)}  w*ĉ²={contrib:.6f}  ĉ={c:.4f}")