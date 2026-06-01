# TODO List - June 1st

## 1. Numerical Support for Current Results
- [ ] Implement a function to calculate MMD (or $MMD^2$) between two probability distributions for a given $\sigma$.
- [ ] Implement Renyi-2 entropy calculation or the $\alpha$ parameter from Joakim's notes.
- [x] Generate a distribution given a target entropy level (Done).
- [ ] Implement a function to find the closest factorized probability distribution.
- [ ] Generate the final plot showing MMD between the two distributions as a function of $\sigma$ and entropy.

## 2. Covariance Matching
- [ ] Explaining/developing covariance matching using Chow-Liu trees.

---

### Notes & Observations
- **BMS:** It is unlikely to produce valid (or nearly valid) probability distributions; consider dropping.
- **IQP Vulnerabilities:** Explore IQP characteristics (e.g., connectivity) that make them "spoofable."