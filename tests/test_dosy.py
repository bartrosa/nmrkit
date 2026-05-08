# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from nmrkit.dosy import (
    DOSYExperiment,
    fit_diffusion_coefficient,
    fit_diffusion_for_peaks,
    load_dosy,
)
from nmrkit.kinetics.nmf import KineticComponents
from nmrkit.kinetics.refine_with_dosy import refine_clusters_with_dosy
from nmrkit.peaks.detect import Peak2D
from nmrkit.peaks.track import Trajectory


def _make_traj(tid: int, curve: np.ndarray) -> Trajectory:
    n = int(curve.shape[0])
    return Trajectory(
        trajectory_id=tid,
        frame_indices=np.arange(n, dtype=np.int_),
        peaks=[],
        _intensity_curve_full=np.asarray(curve, dtype=np.float64),
    )


def test_synthetic_dosy_two_peaks_recover_D_within_tolerance(tmp_path) -> None:
    """Two isolated direct columns decay with distinct D along gradient dimension."""
    n_grad = 32
    n_dir = 400
    direct_ppm = np.linspace(0.0, 10.0, n_dir)
    idx1 = int(np.argmin(np.abs(direct_ppm - 1.0)))
    idx2 = int(np.argmin(np.abs(direct_ppm - 6.0)))

    gradient_strengths_T_per_m = np.linspace(0.002, 0.08, n_grad)
    gamma_rad_per_T_per_s = 2.67522212e8
    delta_s = 2e-3
    Delta_s = 50e-3
    b = (
        (gamma_rad_per_T_per_s**2)
        * (gradient_strengths_T_per_m**2)
        * (delta_s**2)
        * (Delta_s - delta_s / 3.0)
    )

    D1 = 2.5e-10
    D2 = 8.0e-11
    data = np.zeros((n_grad, n_dir), dtype=np.float64)
    for i in range(n_grad):
        data[i, idx1] = 100.0 * np.exp(-D1 * b[i])
        data[i, idx2] = 50.0 * np.exp(-D2 * b[i])

    p = tmp_path / "dosy.npz"
    np.savez(
        p,
        data=data,
        direct_ppm=direct_ppm,
        gradient_strengths_T_per_m=gradient_strengths_T_per_m,
        delta_s=delta_s,
        Delta_s=Delta_s,
        gamma_rad_per_T_per_s=gamma_rad_per_T_per_s,
    )

    exp = load_dosy(p)
    assert isinstance(exp, DOSYExperiment)

    d1, se1 = fit_diffusion_coefficient(exp, 1.0, integration_window_ppm=0.05)
    d2, se2 = fit_diffusion_coefficient(exp, 6.0, integration_window_ppm=0.05)

    assert abs(d1 - D1) / D1 < 0.05
    assert abs(d2 - D2) / D2 < 0.05
    assert np.isfinite(se1) and se1 >= 0.0
    assert np.isfinite(se2) and se2 >= 0.0

    pk = [
        Peak2D(direct_ppm=1.0, indirect_ppm=0.0, intensity=1.0, direct_idx=0, indirect_idx=0),
        Peak2D(direct_ppm=6.0, indirect_ppm=0.0, intensity=1.0, direct_idx=0, indirect_idx=0),
    ]
    out = fit_diffusion_for_peaks(exp, pk, integration_window_ppm=0.05)
    assert len(out) == 2
    assert abs(out[0][0] - D1) / D1 < 0.05
    assert abs(out[1][0] - D2) / D2 < 0.05


def test_refine_clusters_splits_by_diffusion() -> None:
    """Same kinetic cluster and similar kinetics, two diffusion groups → two labels."""
    n_time = 15
    curve = np.exp(-np.linspace(0.0, 3.0, n_time) / 2.0).astype(np.float64)
    trajs = [_make_traj(i, curve.copy()) for i in range(4)]

    components = KineticComponents(
        n_components=1,
        W=np.ones((4, 1), dtype=np.float64),
        H=np.ones((1, n_time), dtype=np.float64),
        trajectory_ids=np.array([0, 1, 2, 3], dtype=np.int_),
        cluster_assignments=np.zeros(4, dtype=np.int_),
    )
    diffusion_per_trajectory = {
        0: (1.0e-10, 1e-12),
        1: (1.0001e-10, 1e-12),
        2: (5.0e-10, 1e-12),
        3: (5.0001e-10, 1e-12),
    }

    refined = refine_clusters_with_dosy(
        components,
        trajs,
        diffusion_per_trajectory,
        log10_d_tolerance=0.15,
    )

    assert refined.shape == (4,)
    assert refined[0] == refined[1]
    assert refined[2] == refined[3]
    assert refined[0] != refined[2]
    assert len(np.unique(refined)) == 2
