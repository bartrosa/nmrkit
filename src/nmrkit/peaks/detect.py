# SPDX-License-Identifier: Apache-2.0

"""Classical 2D peak picking (local maxima + optional parabolic centroid refinement)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter

from nmrkit.io.nmr import NMRSpectrum2D, NMRTimeSeries


def _fractional_index_to_ppm(idx: float, ppm_axis: NDArray[np.float64]) -> float:
    """Linear interpolation from fractional index to ppm (axis aligned with ``spectrum.data``)."""
    n = int(ppm_axis.shape[0])
    if n == 0:
        return float("nan")
    if n == 1:
        return float(ppm_axis[0])
    idx_clamped = float(np.clip(idx, 0.0, float(n - 1)))
    i0 = int(np.floor(idx_clamped))
    i1 = min(i0 + 1, n - 1)
    t = idx_clamped - float(i0)
    return float(ppm_axis[i0] * (1.0 - t) + ppm_axis[i1] * t)


def _parabolic_offset(left: float, center: float, right: float, eps: float = 1e-12) -> float:
    """Sub-pixel offset (in index units) from a 3-point parabolic maximum along one axis."""
    denom = left - 2.0 * center + right
    if abs(denom) < eps:
        return 0.0
    return 0.5 * (left - right) / denom


def _refine_indices(
    data: NDArray[np.float64],
    i_ind: int,
    i_dir: int,
) -> tuple[float, float]:
    """Return fractional indices (indirect, direct); fallback to integers on edges."""
    n0, n1 = data.shape
    di = 0.0
    dj = 0.0
    if 1 <= i_ind <= n0 - 2:
        col = data[i_ind - 1 : i_ind + 2, i_dir]
        di = _parabolic_offset(float(col[0]), float(col[1]), float(col[2]))
    if 1 <= i_dir <= n1 - 2:
        row = data[i_ind, i_dir - 1 : i_dir + 2]
        dj = _parabolic_offset(float(row[0]), float(row[1]), float(row[2]))
    return float(i_ind) + di, float(i_dir) + dj


@dataclass(frozen=True)
class Peak2D:
    direct_ppm: float
    indirect_ppm: float
    intensity: float
    direct_idx: int
    indirect_idx: int


def detect_peaks(
    spectrum: NMRSpectrum2D,
    threshold_rel: float = 0.02,
    min_separation_pts: int = 3,
    refine_centroid: bool = True,
    polarity: Literal["positive", "negative", "both"] = "positive",
) -> list[Peak2D]:
    """Detect local maxima in a 2D spectrum using a footprinted maximum filter.

    Peaks on the outermost rows/columns are still returned when they pass the
    threshold and local-maximum test; **sub-pixel refinement is skipped** on any
    axis where the 3-point neighborhood would extend outside the matrix (corners
    and edges), so ``direct_ppm`` / ``indirect_ppm`` match the discrete axis at
    the integer pixel in those cases.
    """
    data = np.asarray(spectrum.data, dtype=np.float64)
    if data.ndim != 2:
        msg = f"Expected 2D data, got shape {data.shape}"
        raise ValueError(msg)
    if threshold_rel < 0:
        msg = "threshold_rel must be non-negative"
        raise ValueError(msg)
    if min_separation_pts < 1:
        msg = "min_separation_pts must be >= 1"
        raise ValueError(msg)

    abs_max = float(np.max(np.abs(data)))
    if abs_max == 0.0:
        return []

    thresh = threshold_rel * abs_max
    footprint = (int(min_separation_pts), int(min_separation_pts))

    ind_ppm = np.asarray(spectrum.indirect_ppm, dtype=np.float64)
    dir_ppm = np.asarray(spectrum.direct_ppm, dtype=np.float64)

    peak_mask = np.zeros(data.shape, dtype=bool)

    if polarity in ("positive", "both"):
        work_pos = np.where(data > 0.0, data, -np.inf)
        mx_pos = maximum_filter(work_pos, size=footprint, mode="nearest")
        peak_mask |= (work_pos == mx_pos) & (np.abs(data) > thresh) & (data > 0.0)

    if polarity in ("negative", "both"):
        work_neg = np.where(data < 0.0, -data, -np.inf)
        mx_neg = maximum_filter(work_neg, size=footprint, mode="nearest")
        peak_mask |= (work_neg == mx_neg) & (np.abs(data) > thresh) & (data < 0.0)

    idxs = np.argwhere(peak_mask)
    peaks: list[Peak2D] = []
    for i_ind, i_dir in idxs:
        ii = int(i_ind)
        jj = int(i_dir)
        intensity = float(data[ii, jj])

        if refine_centroid:
            fi, fj = _refine_indices(data, ii, jj)
            i_ppm = _fractional_index_to_ppm(fi, ind_ppm)
            d_ppm = _fractional_index_to_ppm(fj, dir_ppm)
        else:
            i_ppm = _fractional_index_to_ppm(float(ii), ind_ppm)
            d_ppm = _fractional_index_to_ppm(float(jj), dir_ppm)

        peaks.append(
            Peak2D(
                direct_ppm=d_ppm,
                indirect_ppm=i_ppm,
                intensity=intensity,
                direct_idx=jj,
                indirect_idx=ii,
            )
        )

    return peaks


def detect_peaks_in_series(series: NMRTimeSeries, **kwargs: Any) -> list[list[Peak2D]]:
    """Run :func:`detect_peaks` on each spectrum in order (no cross-frame tracking)."""
    return [detect_peaks(spec, **kwargs) for spec in series.spectra]
