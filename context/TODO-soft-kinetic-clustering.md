# Soft kinetic clustering (deferred)

Hard assignment via `argmax(W, axis=1)` discards uncertainty when loadings are diffuse or multimodal.

## Possible upgrades

1. **Gaussian mixture on rows of `W`**  
   Treat each peak loading vector as a feature vector in \(\mathbb{R}^k_+\) (after optional normalization). Fit a GMM with Dirichlet priors or simple full-covariance GMM in log-space; posterior responsibilities give **soft** cluster membership and principled model selection (BIC).

2. **Constrained / structured NMF**  
   - **Smooth `H`**: penalize \(\|\Delta H\|_F^2\) or TV along time so component curves are kinetic-looking and less noisy.  
   - **Sparse `W`**: \(L_1\) on rows or columns to encourage each peak to load on few components (parsimony).  
   - **Coupled factorization** if multiple experiments share components.

3. **Combined pipeline**  
   NMF for a nonnegative basis, then GMM on `W` for probabilistic grouping; or variational NMF with hierarchical priors.

These are **not** implemented in the baseline `fit_kinetic_components`; validate the simple NMF baseline on real data before adding complexity.
