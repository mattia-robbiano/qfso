# TODO List - June 1st

## 1. Numerical Support for Current Results
- [x] Implement a function to calculate exact MMD (or $MMD^2$) between two probability distributions for a given $\sigma$.
- [ ] Implement a function to calculate approximate MMD between
two distribution computing partial WH transform keeping only the coefficients where the filter is (based on sigma).
- [x] Implement Renyi-2 entropy calculation or the $\alpha$ parameter from Joakim's notes.
- [x] Generate a distribution given a target entropy level (Done).
- [x] Implement a function to find the closest factorized probability distribution.
- [ ] Generate the final plot showing MMD between the two distributions as a function of $\sigma$ and entropy.
- [ ] 

## 2. Covariance Matching
- [ ] Explaining/developing covariance matching using Chow-Liu trees.

## 3. Extra
- [ ] Class for computing full wh transform given a generic probability vector. Look on SciPy for Fast-WHT (FWHT) and in that case maybe do a wrapper.
---

### Notes & Observations
- **BMS:** It is unlikely to produce valid (or nearly valid) probability distributions; consider dropping.
- **IQP Vulnerabilities:** Explore IQP characteristics (e.g., connectivity) that make them "spoofable."