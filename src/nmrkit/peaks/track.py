# SPDX-License-Identifier: Apache-2.0

"""Baseline peak tracking across time-series frames (pairwise Hungarian + chaining)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from nmrkit.peaks.detect import Peak2D

_BIG_COST = 1e18
_EPS = 1e-30


def _finite_ratio(num: float, den: float) -> float:
    return (abs(float(num)) + _EPS) / (abs(float(den)) + _EPS)


def _pair_cost(
    source: Peak2D,
    dest: Peak2D,
    *,
    span_frames: int,
    direct_tol_ppm: float,
    indirect_tol_ppm: float,
    intensity_log_tol: float,
    weight_direct: float,
    weight_indirect: float,
    weight_intensity: float,
) -> float:
    """Quadratic cost; returns ``_BIG_COST`` if any normalized component exceeds 1."""
    if span_frames < 1:
        return _BIG_COST
    span = float(span_frames)
    dd = abs(float(dest.direct_ppm) - float(source.direct_ppm)) / (float(direct_tol_ppm) * span)
    di = abs(float(dest.indirect_ppm) - float(source.indirect_ppm)) / (
        float(indirect_tol_ppm) * span
    )
    ratio = _finite_ratio(dest.intensity, source.intensity)
    li = abs(float(np.log10(ratio))) / (float(intensity_log_tol) * span)
    if dd > 1.0 or di > 1.0 or li > 1.0:
        return _BIG_COST
    return (
        float(weight_direct) * dd * dd
        + float(weight_indirect) * di * di
        + float(weight_intensity) * li * li
    )


@dataclass(frozen=True)
class Trajectory:
    """A peak followed across multiple frames."""

    trajectory_id: int
    frame_indices: NDArray[np.int_]
    peaks: list[Peak2D]
    _intensity_curve_full: NDArray[np.float64] = field(repr=False)

    @property
    def direct_ppm_mean(self) -> float:
        if not self.peaks:
            return float("nan")
        return float(np.mean([p.direct_ppm for p in self.peaks]))

    @property
    def indirect_ppm_mean(self) -> float:
        if not self.peaks:
            return float("nan")
        return float(np.mean([p.indirect_ppm for p in self.peaks]))

    @property
    def intensity_curve(self) -> NDArray[np.float64]:
        """Intensity vs frame index; length ``n_total_frames`` from :func:`track_peaks`."""
        return self._intensity_curve_full


def track_peaks(
    peaks_per_frame: list[list[Peak2D]],
    direct_tol_ppm: float = 0.05,
    indirect_tol_ppm: float = 0.5,
    intensity_log_tol: float = 1.0,
    weight_direct: float = 1.0,
    weight_indirect: float = 0.1,
    weight_intensity: float = 0.5,
    allow_gap_frames: int = 1,
    n_total_frames: int | None = None,
) -> list[Trajectory]:
    """Chain peaks across frames using pairwise Hungarian assignment with optional gaps."""
    if direct_tol_ppm <= 0 or indirect_tol_ppm <= 0:
        msg = "Tolerance parameters must be positive."
        raise ValueError(msg)
    if intensity_log_tol <= 0:
        msg = "intensity_log_tol must be positive."
        raise ValueError(msg)
    if allow_gap_frames < 0:
        msg = "allow_gap_frames must be non-negative."
        raise ValueError(msg)

    n_frames = len(peaks_per_frame)
    if n_frames == 0:
        return []

    n_tot = int(n_total_frames) if n_total_frames is not None else n_frames
    if n_tot < n_frames:
        msg = "n_total_frames must be >= len(peaks_per_frame)."
        raise ValueError(msg)

    traj_events: dict[int, list[tuple[int, Peak2D]]] = {}
    peak_traj_ids: list[list[int]] = [[] for _ in range(n_frames)]
    next_traj_id = 0

    # Trajectories that missed an assignment at the last opportunity need a gap row,
    # not a duplicate consecutive row, on subsequent frames.
    pending_map: dict[int, tuple[Peak2D, int]] = {}

    for _i, pk in enumerate(peaks_per_frame[0]):
        tid = next_traj_id
        next_traj_id += 1
        traj_events[tid] = [(0, pk)]
        peak_traj_ids[0].append(tid)

    for dest in range(1, n_frames):
        dest_peaks = peaks_per_frame[dest]

        # Drop hopeless pending rows (too many skipped frames before we reach dest).
        for tid, (_pk, sf) in list(pending_map.items()):
            skipped_between = dest - sf - 1
            if skipped_between > allow_gap_frames:
                del pending_map[tid]

        sources: list[tuple[Peak2D, int, int]] = []

        # Gap-bridging rows (always ahead of consecutive rows for the same tid).
        for tid, (pk, sf) in pending_map.items():
            skipped_between = dest - sf - 1
            if skipped_between < 1:
                continue
            if skipped_between > allow_gap_frames:
                continue
            sources.append((pk, tid, sf))

        for idx, pk in enumerate(peaks_per_frame[dest - 1]):
            tid = peak_traj_ids[dest - 1][idx]
            if tid in pending_map:
                continue
            sources.append((pk, tid, dest - 1))

        n_src = len(sources)
        n_dst = len(dest_peaks)
        new_dest_traj = [-1] * n_dst

        if n_dst == 0:
            for pk, tid, src_frame in sources:
                pending_map[tid] = (pk, src_frame)
            peak_traj_ids[dest] = []
            continue

        if n_src == 0:
            for j in range(n_dst):
                tid = next_traj_id
                next_traj_id += 1
                traj_events[tid] = [(dest, dest_peaks[j])]
                new_dest_traj[j] = tid
            peak_traj_ids[dest] = new_dest_traj
            continue

        cost = np.full((n_src, n_dst), _BIG_COST, dtype=np.float64)
        for r, (spk, _tid, src_frame) in enumerate(sources):
            span = dest - src_frame
            for c in range(n_dst):
                cost[r, c] = _pair_cost(
                    spk,
                    dest_peaks[c],
                    span_frames=int(span),
                    direct_tol_ppm=direct_tol_ppm,
                    indirect_tol_ppm=indirect_tol_ppm,
                    intensity_log_tol=intensity_log_tol,
                    weight_direct=weight_direct,
                    weight_indirect=weight_indirect,
                    weight_intensity=weight_intensity,
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_rows: set[int] = set()
        matched_cols: set[int] = set()
        for r, c in zip(row_ind, col_ind, strict=False):
            if cost[r, c] >= _BIG_COST / 2.0:
                continue
            matched_rows.add(int(r))
            matched_cols.add(int(c))
            _spk, tid, _sf = sources[r]
            pk = dest_peaks[c]
            traj_events.setdefault(tid, []).append((dest, pk))
            new_dest_traj[c] = tid
            pending_map.pop(tid, None)

        for c in range(n_dst):
            if c in matched_cols:
                continue
            tid = next_traj_id
            next_traj_id += 1
            traj_events[tid] = [(dest, dest_peaks[c])]
            new_dest_traj[c] = tid

        for r, (pk, tid, src_frame) in enumerate(sources):
            if r in matched_rows:
                continue
            if src_frame == dest - 1:
                pending_map[tid] = (pk, src_frame)

        peak_traj_ids[dest] = new_dest_traj

    out: list[Trajectory] = []
    for tid in sorted(traj_events.keys()):
        events = sorted(traj_events[tid], key=lambda x: x[0])
        frames_arr = np.array([f for f, _ in events], dtype=np.int_)
        peaks_list = [p for _, p in events]
        curve = np.full(n_tot, np.nan, dtype=np.float64)
        for fi, pk in zip(frames_arr.tolist(), peaks_list, strict=False):
            if 0 <= fi < n_tot:
                curve[int(fi)] = float(pk.intensity)
        out.append(
            Trajectory(
                trajectory_id=tid,
                frame_indices=frames_arr,
                peaks=peaks_list,
                _intensity_curve_full=curve,
            )
        )

    return out
