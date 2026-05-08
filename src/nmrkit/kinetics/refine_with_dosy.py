# SPDX-License-Identifier: Apache-2.0

"""Refine kinetic NMF clusters using diffusion coefficients."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nmrkit.kinetics.nmf import KineticComponents
from nmrkit.peaks.track import Trajectory


def _uf_find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _uf_union(parent: list[int], i: int, j: int) -> None:
    ri = _uf_find(parent, i)
    rj = _uf_find(parent, j)
    if ri != rj:
        parent[rj] = ri


def refine_clusters_with_dosy(
    components: KineticComponents,
    trajectories: list[Trajectory],
    diffusion_per_trajectory: dict[int, tuple[float, float]],
    log10_d_tolerance: float = 0.15,
) -> NDArray[np.int_]:
    """Split kinetic clusters by similar diffusion (single-linkage on :math:`\\log_{10} D`).

    Within each original kinetic cluster, trajectories whose ``log10(D)`` differs by
    at most ``log10_d_tolerance`` are merged (single linkage). New integer labels are
    assigned starting above ``max(cluster_assignments)``; trajectories without a
    positive ``D`` entry keep their original kinetic label.
    """
    n = len(trajectories)
    if components.cluster_assignments.shape[0] != n:
        msg = "cluster_assignments length must match trajectories."
        raise ValueError(msg)
    if components.trajectory_ids.shape[0] != n:
        msg = "trajectory_ids length must match trajectories."
        raise ValueError(msg)

    refined = np.asarray(components.cluster_assignments, dtype=np.int_).copy()
    next_label = int(np.max(components.cluster_assignments)) + 1

    for k in np.unique(components.cluster_assignments):
        idxs = np.flatnonzero(components.cluster_assignments == k)
        members: list[tuple[int, float]] = []
        for i in idxs.tolist():
            tid = int(trajectories[i].trajectory_id)
            if tid not in diffusion_per_trajectory:
                continue
            d_val, _stderr = diffusion_per_trajectory[tid]
            if d_val <= 0.0 or not np.isfinite(d_val):
                continue
            members.append((int(i), float(d_val)))

        if len(members) <= 1:
            continue

        mloc = len(members)
        parent = list(range(mloc))
        logd = [np.log10(d) for _i, d in members]
        for i in range(mloc):
            for j in range(i + 1, mloc):
                if abs(logd[i] - logd[j]) <= float(log10_d_tolerance):
                    _uf_union(parent, i, j)

        roots = {_uf_find(parent, i) for i in range(mloc)}
        if len(roots) <= 1:
            continue

        root_to_label: dict[int, int] = {}
        for r in sorted(roots):
            root_to_label[r] = next_label
            next_label += 1
        for i in range(mloc):
            gi = members[i][0]
            refined[gi] = root_to_label[_uf_find(parent, i)]

    return refined
