# SPDX-License-Identifier: Apache-2.0

"""DOSY / gradient-series diffusion fitting (Stejskal–Tanner)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from nmrkit.peaks.detect import Peak2D


@dataclass(frozen=True)
class DOSYExperiment:
    """A pseudo-2D DOSY: rows are gradient steps, columns are direct ppm."""

    data: np.ndarray  # (n_gradients, n_direct), real-valued
    direct_ppm: np.ndarray  # 1D, length n_direct
    gradient_strengths_T_per_m: np.ndarray  # 1D, length n_gradients, in T/m
    delta_s: float  # gradient pulse duration (s)
    Delta_s: float  # diffusion delay (s)
    gamma_rad_per_T_per_s: float  # gyromagnetic ratio of observed nucleus (rad·s⁻¹·T⁻¹)


def load_dosy(path: Path) -> DOSYExperiment:
    """Load a DOSY experiment from a NumPy ``.npz`` archive.

    Required keys: ``data``, ``direct_ppm``, ``gradient_strengths_T_per_m``,
    ``delta_s``, ``Delta_s``, ``gamma_rad_per_T_per_s``.
    Scalar parameters may be 0-d arrays.
    """
    p = Path(path)
    if p.suffix.lower() != ".npz":
        msg = f"Expected a .npz file, got {p}"
        raise ValueError(msg)
    z = np.load(p, allow_pickle=False)
    required = (
        "data",
        "direct_ppm",
        "gradient_strengths_T_per_m",
        "delta_s",
        "Delta_s",
        "gamma_rad_per_T_per_s",
    )
    for k in required:
        if k not in z:
            msg = f"Missing key {k!r} in {p}"
            raise ValueError(msg)

    def _scalar(name: str) -> float:
        a = z[name]
        return float(np.asarray(a).reshape(-1)[0])

    return DOSYExperiment(
        data=np.asarray(z["data"], dtype=np.float64),
        direct_ppm=np.asarray(z["direct_ppm"], dtype=np.float64),
        gradient_strengths_T_per_m=np.asarray(z["gradient_strengths_T_per_m"], dtype=np.float64),
        delta_s=_scalar("delta_s"),
        Delta_s=_scalar("Delta_s"),
        gamma_rad_per_T_per_s=_scalar("gamma_rad_per_T_per_s"),
    )


def _b_values_stejskal_tanner(
    G: NDArray[np.float64],
    *,
    gamma: float,
    delta: float,
    Delta: float,
) -> NDArray[np.float64]:
    """Stejskal–Tanner b-factor (s/m²) for rectangular gradient pulses."""
    g = np.asarray(G, dtype=np.float64)
    # b = γ² G² δ² (Δ − δ/3)
    return (gamma**2) * (g**2) * (delta**2) * (Delta - delta / 3.0)


def _integration_mask(
    direct_ppm: NDArray[np.float64],
    center_ppm: float,
    window_ppm: float,
) -> NDArray[np.bool_]:
    half = float(window_ppm) / 2.0
    return np.abs(direct_ppm - float(center_ppm)) <= half


def fit_diffusion_coefficient(
    experiment: DOSYExperiment,
    direct_ppm: float,
    integration_window_ppm: float = 0.05,
) -> tuple[float, float]:
    """Return ``(D, D_stderr)`` in m²/s (Stejskal–Tanner exponential decay in ``b``)."""
    dmat = np.asarray(experiment.data, dtype=np.float64)
    if dmat.ndim != 2:
        msg = f"Expected 2D data, got shape {dmat.shape}"
        raise ValueError(msg)
    n_grad, n_dir = dmat.shape
    ppm = np.asarray(experiment.direct_ppm, dtype=np.float64)
    if ppm.shape[0] != n_dir:
        msg = "direct_ppm length must match data columns."
        raise ValueError(msg)

    g = np.asarray(experiment.gradient_strengths_T_per_m, dtype=np.float64).ravel()
    if g.shape[0] != n_grad:
        msg = "gradient_strengths length must match data rows."
        raise ValueError(msg)

    mask = _integration_mask(ppm, direct_ppm, integration_window_ppm)
    if not np.any(mask):
        msg = "Integration window does not overlap any direct ppm column."
        raise ValueError(msg)

    y = np.sum(dmat[:, mask], axis=1)
    y_pos = np.maximum(y, 1e-30)

    b = _b_values_stejskal_tanner(
        g,
        gamma=float(experiment.gamma_rad_per_T_per_s),
        delta=float(experiment.delta_s),
        Delta=float(experiment.Delta_s),
    )

    log_y = np.log(y_pos)
    res = stats.linregress(np.asarray(b, dtype=np.float64), np.asarray(log_y, dtype=np.float64))
    d_est = float(-res.slope)
    d_stderr = float(res.stderr if res.stderr is not None else np.nan)
    if not np.isfinite(d_stderr):
        d_stderr = 0.0
    return d_est, d_stderr


def fit_diffusion_for_peaks(
    experiment: DOSYExperiment,
    peaks: list[Peak2D],
    integration_window_ppm: float = 0.05,
) -> dict[int, tuple[float, float]]:
    """Map peak list index → ``(D, D_stderr)`` using each peak's ``direct_ppm`` center."""
    out: dict[int, tuple[float, float]] = {}
    for idx, pk in enumerate(peaks):
        out[idx] = fit_diffusion_coefficient(
            experiment,
            float(pk.direct_ppm),
            integration_window_ppm=integration_window_ppm,
        )
    return out
