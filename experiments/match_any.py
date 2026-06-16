import numpy as np
from qfso.distributions import discretized_normal_probability, plot_distributions, uniform_like
from qfso.models.statevector import (
    hamming_weight,
    match_first_order,
    match_mmd_optimal,
    mmd_squared,
    sorted_mmd_contributions,
)


if __name__ == "__main__":
    n = 10
    p = discretized_normal_probability((-1, 10), 2**n)
    u = uniform_like(p)

    sigma = 0.1 * n
    q = match_mmd_optimal(p, sigma=sigma, hw_min=1, hw_max=3)
    f = np.asarray(match_first_order(p))

    mmd_pu = mmd_squared(p, u, sigma)
    mmd_pf = mmd_squared(p, f, sigma)
    mmd_pq = mmd_squared(p, q, sigma)
    print(f"MMD²(p, uniform) = {mmd_pu:.6f}")
    print(f"MMD²(p, first-order) = {mmd_pf:.6f}")
    print(f"MMD²(p, q)       = {mmd_pq:.6f}")

    plot_distributions(
        [p, q, u, f],
        labels=["target p", f"factorized q (mmd={mmd_pq:.4e})", f"uniform       (mmd={mmd_pu:.4e})", f"first-order   (mmd={mmd_pf:.4e})"],
        title="MMD-optimal factorized approximation",
    )

    contributions = sorted_mmd_contributions(p, sigma=sigma, hw_min=1, hw_max=3)
    print("\nTop MMD contributions:")
    for k, contrib, c in contributions[:10]:
        print(f"  k={k:0{n}b}  hw={hamming_weight(k)}  w*ĉ²={contrib:.6f}  ĉ={c:.4f}")


    #TODO compute mmd exact and model 500 shots

