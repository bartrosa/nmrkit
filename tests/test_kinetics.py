# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from nmrkit.kinetics.nmf import fit_kinetic_components
from nmrkit.peaks.track import Trajectory


def _make_traj(tid: int, curve: np.ndarray) -> Trajectory:
    n = int(curve.shape[0])
    return Trajectory(
        trajectory_id=tid,
        frame_indices=np.arange(n, dtype=np.int_),
        peaks=[],
        _intensity_curve_full=np.asarray(curve, dtype=np.float64),
    )


def _decay_growth_templates(n_time: int = 20) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 10.0, n_time)
    decay = np.exp(-t / 4.0)
    growth = 1.0 - np.exp(-t / 4.0)
    return decay.astype(np.float64), growth.astype(np.float64)


def test_nmf_two_kinetic_groups_and_shape_recovery() -> None:
    decay, growth = _decay_growth_templates(20)
    rng = np.random.default_rng(42)

    curves = []
    for _ in range(3):
        curves.append(decay + rng.normal(0.0, 0.02, size=decay.shape))
    for _ in range(3):
        curves.append(growth + rng.normal(0.0, 0.02, size=growth.shape))

    trajs = [_make_traj(i, curves[i]) for i in range(6)]

    kin = fit_kinetic_components(
        trajs,
        n_components=2,
        normalize="max",
        impute_missing="interp",
        random_state=0,
    )

    _, counts = np.unique(kin.cluster_assignments, return_counts=True)
    assert sorted(counts.tolist()) == [3, 3]

    h = kin.H
    decay_n = (decay - decay.mean()) / (decay.std() + 1e-9)
    growth_n = (growth - growth.mean()) / (growth.std() + 1e-9)

    best_decay = max(float(np.corrcoef(decay_n, row)[0, 1]) for row in h)
    best_growth = max(float(np.corrcoef(growth_n, row)[0, 1]) for row in h)
    assert best_decay > 0.95
    assert best_growth > 0.95


def test_auto_k_selects_two_components() -> None:
    decay, growth = _decay_growth_templates(20)
    rng = np.random.default_rng(7)
    curves = [decay + rng.normal(0.0, 0.02, size=decay.shape) for _ in range(3)]
    curves += [growth + rng.normal(0.0, 0.02, size=growth.shape) for _ in range(3)]
    trajs = [_make_traj(i, curves[i]) for i in range(6)]

    kin = fit_kinetic_components(
        trajs,
        n_components="auto",
        n_components_max=6,
        normalize="max",
        impute_missing="interp",
        random_state=1,
    )
    assert kin.n_components == 2


def test_interp_after_random_nan_preserves_clusters() -> None:
    decay, growth = _decay_growth_templates(20)
    rng = np.random.default_rng(99)
    curves = [decay + rng.normal(0.0, 0.02, size=decay.shape) for _ in range(3)]
    curves += [growth + rng.normal(0.0, 0.02, size=growth.shape) for _ in range(3)]

    drop_idx = [3, 14]
    for c in curves:
        for j in drop_idx:
            c[j] = np.nan

    trajs = [_make_traj(i, curves[i]) for i in range(6)]

    kin = fit_kinetic_components(
        trajs,
        n_components=2,
        normalize="max",
        impute_missing="interp",
        random_state=2,
    )
    _, counts = np.unique(kin.cluster_assignments, return_counts=True)
    assert sorted(counts.tolist()) == [3, 3]
