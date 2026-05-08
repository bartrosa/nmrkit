# Joint multi-frame tracking (deferred upgrade)

The baseline in `nmrkit.peaks.track` chains **pairwise Hungarian assignments** between consecutive frames (with a small pending queue for short gaps). This is fast and deterministic, but it is **greedy**: early mismatches cannot be corrected once frames advance.

## Upgrade path

1. **Differentiable assignment relaxation**  
   Replace hard Hungarian decisions with a **Sinkhorn / optimal transport** layer (entropy-regularized assignment) between frames or small windows. This yields smooth gradients for downstream optimization and soft correspondence matrices instead of brittle one-shot minima.

2. **Learned peak descriptors**  
   Augment classical ppm/intensity features with a lightweight encoder (e.g. small CNN or MLP on a local spectral patch, or hand-crafted moments + embedding). Learned features better separate overlapping ridges and symmetric artefacts than ppm drift alone.

3. **Joint optimization across all frames**  
   Instead of chaining \(T-1\) independent matchings, optimize trajectories **globally**: variables include assignment tensors between all adjacent (or all) frame pairs with cycle-consistency or temporal smoothness penalties on ppm and intensity. Possible frameworks: continuous relaxation + gradient descent, ADMM on assignment matrices, or ILP for small \(T\).

4. **Evaluation & constraints**  
   Ground-truth from simulation or expert annotations; metrics such as purity, completeness, and longitudinal RMS ppm drift. Optionally enforce **physics-inspired priors** (smooth intensity decay, bounds on linewidth proxies).

This module is intentionally **not** implemented yet; profiling on real kinetic HSQC stacks should justify the added complexity over the pairwise baseline.
