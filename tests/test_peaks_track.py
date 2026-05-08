# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from nmrkit.peaks.detect import Peak2D
from nmrkit.peaks.track import track_peaks


def _pk(d: float, i: float, intensity: float) -> Peak2D:
    return Peak2D(
        direct_ppm=d,
        indirect_ppm=i,
        intensity=intensity,
        direct_idx=0,
        indirect_idx=0,
    )


def test_three_peaks_four_frames_smooth_drift() -> None:
    common_kw = dict(
        direct_tol_ppm=0.2,
        indirect_tol_ppm=3.0,
        intensity_log_tol=2.0,
        weight_direct=1.0,
        weight_indirect=0.1,
        weight_intensity=0.5,
        allow_gap_frames=0,
    )

    peaks_per_frame = []
    for t in range(4):
        frame = [
            _pk(8.0 + 0.02 * t, 120.0 + 0.3 * t, 10.0 - 0.1 * t),
            _pk(7.2 + 0.015 * t, 95.0 + 0.2 * t, 9.0),
            _pk(6.5 + 0.01 * t, 70.0 + 0.25 * t, 8.5 + 0.05 * t),
        ]
        peaks_per_frame.append(frame)

    trajs = track_peaks(peaks_per_frame, **common_kw)
    assert len(trajs) == 3
    assert all(len(tr.peaks) == 4 for tr in trajs)


def test_new_peak_mid_series_starts_new_trajectory() -> None:
    """Peak appears from frame index 2 onward → trajectory length 3 on frames 2–4."""
    peaks_per_frame = [
        [],
        [],
        [_pk(7.0, 50.0, 5.0)],
        [_pk(7.01, 50.1, 5.2)],
        [_pk(7.02, 50.15, 5.4)],
    ]
    trajs = track_peaks(
        peaks_per_frame,
        direct_tol_ppm=0.2,
        indirect_tol_ppm=3.0,
        intensity_log_tol=2.0,
        allow_gap_frames=0,
    )
    lengths = sorted(len(tr.peaks) for tr in trajs)
    assert lengths[-1] == 3
    starter = next(tr for tr in trajs if len(tr.peaks) == 3)
    assert list(starter.frame_indices) == [2, 3, 4]


def test_peak_vanishes_trajectory_stops() -> None:
    """Substrate in frames 0–1 only; later frames only contain a distant unrelated peak."""
    peaks_per_frame = [
        [_pk(8.0, 120.0, 10.0)],
        [_pk(8.01, 119.8, 9.8)],
        [_pk(3.0, 20.0, 7.0)],
        [_pk(3.1, 21.0, 7.2)],
    ]
    trajs = track_peaks(
        peaks_per_frame,
        direct_tol_ppm=0.05,
        indirect_tol_ppm=0.5,
        intensity_log_tol=1.5,
        allow_gap_frames=0,
    )
    short = [tr for tr in trajs if len(tr.peaks) == 2]
    assert len(short) == 1
    assert list(short[0].frame_indices) == [0, 1]


def test_gap_allow_bridges_missing_frame() -> None:
    peaks_per_frame = [
        [_pk(7.0, 60.0, 10.0)],
        [],
        [_pk(7.02, 60.2, 9.8)],
        [_pk(7.03, 60.25, 9.7)],
    ]

    trajs_gap = track_peaks(
        peaks_per_frame,
        direct_tol_ppm=0.2,
        indirect_tol_ppm=3.0,
        intensity_log_tol=2.0,
        allow_gap_frames=1,
    )
    max_len_gap = max(len(tr.peaks) for tr in trajs_gap)
    assert max_len_gap == 3

    trajs_no_gap = track_peaks(
        peaks_per_frame,
        direct_tol_ppm=0.2,
        indirect_tol_ppm=3.0,
        intensity_log_tol=2.0,
        allow_gap_frames=0,
    )
    max_len_no = max(len(tr.peaks) for tr in trajs_no_gap)
    # Without gap bridging, the frame-0 seed cannot link to later frames, but
    # subsequent detections can still form shorter chains (here: frames 2→3).
    assert max_len_no == 2
    assert max_len_gap > max_len_no


def test_intensity_curve_padding() -> None:
    peaks_per_frame = [
        [_pk(5.0, 40.0, 1.0)],
        [_pk(5.01, 40.1, 2.0)],
    ]
    trajs = track_peaks(peaks_per_frame, allow_gap_frames=0, n_total_frames=6)
    curve = trajs[0].intensity_curve
    assert curve.shape == (6,)
    assert curve[0] == 1.0
    assert curve[1] == 2.0
    assert np.isnan(curve[2])
