# SPDX-License-Identifier: Apache-2.0

"""Matplotlib visualizations for HSQC overlays and kinetic traces."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nmrkit.io.nmr import NMRSpectrum2D, NMRTimeSeries
from nmrkit.kinetics.nmf import KineticComponents
from nmrkit.peaks.track import Trajectory


def plot_hsqc_with_clusters(
    spectrum: NMRSpectrum2D,
    trajectories: list[Trajectory],
    cluster_assignments: np.ndarray,
    ax: Axes | None = None,
    contour_levels: int = 10,
) -> Figure:
    """Contour plot of ``|spectrum.data|`` with tracked peaks colored by cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
    else:
        fig = cast(Figure, ax.figure)

    data = np.asarray(spectrum.data, dtype=np.float64)
    absd = np.abs(data)
    dir_ppm = np.asarray(spectrum.direct_ppm, dtype=np.float64)
    ind_ppm = np.asarray(spectrum.indirect_ppm, dtype=np.float64)
    xx, yy = np.meshgrid(dir_ppm, ind_ppm)

    amax = float(np.max(absd))
    if amax > 0.0:
        pos = absd[absd > 0]
        hi = float(np.max(pos)) if pos.size else amax
        amin = float(np.max(pos) * 1e-4) if pos.size else amax * 1e-4
        amin = max(amin, 1e-30)
        if hi > amin:
            levels = np.geomspace(amin, hi, num=max(2, contour_levels))
        else:
            levels = np.linspace(amin, hi, num=max(2, contour_levels))
        ax.contour(xx, yy, absd, levels=levels, colors="0.45", linewidths=0.7, alpha=0.75)

    ax.set_xlim(float(dir_ppm[0]), float(dir_ppm[-1]))
    ax.set_ylim(float(ind_ppm[-1]), float(ind_ppm[0]))
    ax.set_xlabel(f"{spectrum.direct_nucleus} (ppm)")
    ax.set_ylabel(f"{spectrum.indirect_nucleus} (ppm)")

    assign = np.asarray(cluster_assignments, dtype=np.int_)
    cmap_scatter = plt.get_cmap("tab10")

    for i, tr in enumerate(trajectories):
        cid = int(assign[i]) if i < assign.shape[0] else 0
        color = cmap_scatter(cid % 10)
        ax.scatter(
            [tr.direct_ppm_mean],
            [tr.indirect_ppm_mean],
            s=36,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.8,
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_kinetic_curves(
    components: KineticComponents,
    timestamps: np.ndarray,
    ax: Axes | None = None,
) -> Figure:
    """Plot each kinetic component (rows of ``H``) versus time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
    else:
        fig = cast(Figure, ax.figure)

    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    h = np.asarray(components.H, dtype=np.float64)
    cmap = plt.get_cmap("Set2")
    for k in range(h.shape[0]):
        ax.plot(ts, h[k, :], color=cmap(k % 8), linewidth=2.0, label=f"Component {k}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    return fig


def plot_summary(
    series: NMRTimeSeries,
    trajectories: list[Trajectory],
    components: KineticComponents,
    output_path: Path,
    cluster_assignments: np.ndarray | None = None,
) -> None:
    """Write ``summary.png``: HSQC snapshots, kinetic traces, cluster legend.

    ``output_path`` is the **directory** that will receive ``summary.png``.
    If ``cluster_assignments`` is omitted, ``components.cluster_assignments`` is used.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    assign = (
        np.asarray(cluster_assignments, dtype=np.int_)
        if cluster_assignments is not None
        else np.asarray(components.cluster_assignments, dtype=np.int_)
    )

    n = len(series.spectra)
    if n == 0:
        msg = "Time series has no spectra."
        raise ValueError(msg)

    i0, i1, i2 = 0, n // 2, n - 1
    labels = ("Start", "Mid", "End")

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], width_ratios=[1.0, 1.0, 1.0, 0.35])

    for col, (idx, title) in enumerate(zip((i0, i1, i2), labels, strict=True)):
        ax_hs = fig.add_subplot(gs[0, col])
        plot_hsqc_with_clusters(series.spectra[idx], trajectories, assign, ax=ax_hs)
        ax_hs.set_title(f"{title} (t = {series.timestamps[idx]:.2f} s)")

    ax_k = fig.add_subplot(gs[1, 0:3])
    plot_kinetic_curves(components, series.timestamps, ax=ax_k)
    ax_k.set_title("NMF kinetic components")

    ax_leg = fig.add_subplot(gs[1, 3])
    ax_leg.axis("off")
    uniq = sorted(int(x) for x in np.unique(assign))
    cmap = plt.get_cmap("tab10")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i % 10),
            markersize=9,
            label=f"Cluster {i}",
        )
        for i in uniq
    ]
    ax_leg.legend(handles=handles, loc="upper left", title="Clusters", frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
