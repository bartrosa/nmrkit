# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nmrkit.io.nmr import NMRSpectrum2D, NMRTimeSeries
from nmrkit.kinetics.nmf import KineticComponents
from nmrkit.peaks.detect import Peak2D
from nmrkit.peaks.track import Trajectory
from nmrkit.viz.plot import plot_hsqc_with_clusters, plot_kinetic_curves, plot_summary


def _tiny_spectrum() -> NMRSpectrum2D:
    rng = np.random.default_rng(0)
    data = rng.random((8, 12)) * 0.5
    data[3, 6] = 10.0
    data[5, 9] = 8.0
    indirect_ppm = np.linspace(120.0, 0.0, 8)
    direct_ppm = np.linspace(12.0, 0.0, 12)
    return NMRSpectrum2D(
        data=data,
        direct_ppm=direct_ppm,
        indirect_ppm=indirect_ppm,
        direct_nucleus="1H",
        indirect_nucleus="13C",
        metadata={},
    )


def test_plot_hsqc_with_clusters_returns_figure_with_axes() -> None:
    spec = _tiny_spectrum()
    pk_a = Peak2D(
        direct_ppm=float(spec.direct_ppm[6]),
        indirect_ppm=float(spec.indirect_ppm[3]),
        intensity=1.0,
        direct_idx=6,
        indirect_idx=3,
    )
    pk_b = Peak2D(
        direct_ppm=float(spec.direct_ppm[9]),
        indirect_ppm=float(spec.indirect_ppm[5]),
        intensity=1.0,
        direct_idx=9,
        indirect_idx=5,
    )
    trs = [
        Trajectory(0, np.array([0], dtype=np.int_), [pk_a], np.array([1.0], dtype=np.float64)),
        Trajectory(1, np.array([0], dtype=np.int_), [pk_b], np.array([1.0], dtype=np.float64)),
    ]
    assign = np.array([0, 1], dtype=np.int_)
    fig = plot_hsqc_with_clusters(spec, trs, assign, contour_levels=5)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_kinetic_curves_returns_figure_with_axes() -> None:
    ts = np.linspace(0.0, 10.0, 6)
    h_row0 = np.linspace(1.0, 0.0, 6)
    h_row1 = np.linspace(0.0, 1.0, 6)
    kin = KineticComponents(
        n_components=2,
        W=np.ones((3, 2), dtype=np.float64),
        H=np.vstack([h_row0, h_row1]),
        trajectory_ids=np.arange(3, dtype=np.int_),
        cluster_assignments=np.zeros(3, dtype=np.int_),
    )
    fig = plot_kinetic_curves(kin, ts)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_summary_writes_png(tmp_path) -> None:
    spec = _tiny_spectrum()
    series = NMRTimeSeries(spectra=[spec, spec, spec], timestamps=np.array([0.0, 1.0, 2.0]))
    pk = Peak2D(
        direct_ppm=float(spec.direct_ppm[6]),
        indirect_ppm=float(spec.indirect_ppm[3]),
        intensity=1.0,
        direct_idx=6,
        indirect_idx=3,
    )
    trs = [
        Trajectory(0, np.arange(3, dtype=np.int_), [pk, pk, pk], np.ones(3, dtype=np.float64)),
    ]
    kin = KineticComponents(
        n_components=1,
        W=np.ones((1, 1), dtype=np.float64),
        H=np.ones((1, 3), dtype=np.float64),
        trajectory_ids=np.array([0], dtype=np.int_),
        cluster_assignments=np.zeros(1, dtype=np.int_),
    )
    plot_summary(series, trs, kin, tmp_path)
    assert (tmp_path / "summary.png").is_file()
