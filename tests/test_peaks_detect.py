# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from nmrkit.io.nmr import NMRSpectrum2D, NMRTimeSeries
from nmrkit.peaks.detect import detect_peaks, detect_peaks_in_series


def _frac_idx_to_ppm(idx: float, axis: np.ndarray) -> float:
    """Ground-truth ppm for fractional indices on a uniform ppm axis."""
    ppm = np.asarray(axis, dtype=np.float64)
    n = ppm.shape[0]
    x = np.arange(n, dtype=np.float64)
    return float(np.interp(idx, x, ppm))


def _build_gaussian_spectrum(
    *,
    shape: tuple[int, int],
    centers_idx: list[tuple[float, float, float]],
    noise_frac: float,
    rng_seed: int,
) -> NMRSpectrum2D:
    """centers_idx: (ci, cj, amplitude) in float indices (indirect, direct)."""
    n_ind, n_dir = shape
    ind_ppm = np.linspace(120.0, 10.0, n_ind)
    dir_ppm = np.linspace(14.0, -2.0, n_dir)

    ii, jj = np.indices(shape)
    z = np.zeros(shape, dtype=np.float64)
    for ci, cj, amp in centers_idx:
        z += amp * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * 2.5**2))

    rng = np.random.default_rng(rng_seed)
    z = z + rng.uniform(-noise_frac * z.max(), noise_frac * z.max(), size=z.shape)

    return NMRSpectrum2D(
        data=z,
        direct_ppm=dir_ppm,
        indirect_ppm=ind_ppm,
        direct_nucleus="1H",
        indirect_nucleus="13C",
        metadata={"synthetic": True},
    )


def test_detect_peaks_five_gaussians_recovers_centers() -> None:
    centers = [
        (22.3, 44.7, 10.0),
        (58.1, 91.2, 9.0),
        (95.0, 33.4, 8.5),
        (41.2, 115.6, 8.0),
        (130.5, 78.9, 7.5),
    ]
    spec = _build_gaussian_spectrum(
        shape=(160, 140),
        centers_idx=centers,
        noise_frac=0.01,
        rng_seed=42,
    )

    spacing_ind = abs(float(spec.indirect_ppm[0] - spec.indirect_ppm[1]))
    spacing_dir = abs(float(spec.direct_ppm[0] - spec.direct_ppm[1]))
    tol_ind = 0.5 * spacing_ind
    tol_dir = 0.5 * spacing_dir

    peaks = detect_peaks(
        spec,
        threshold_rel=0.02,
        min_separation_pts=3,
        refine_centroid=True,
        polarity="positive",
    )

    truth_ppm = [
        (_frac_idx_to_ppm(ci, spec.indirect_ppm), _frac_idx_to_ppm(cj, spec.direct_ppm))
        for ci, cj, _amp in centers
    ]

    assert len(peaks) >= len(truth_ppm)

    for ti, td in truth_ppm:
        close = [
            p
            for p in peaks
            if abs(p.indirect_ppm - ti) <= tol_ind and abs(p.direct_ppm - td) <= tol_dir
        ]
        coords = [(p.indirect_ppm, p.direct_ppm) for p in peaks]
        msg = f"no peak near ({ti:.4f}, {td:.4f}) ppm among {coords}"
        assert close, msg


def test_polarity_positive_skips_negative_peaks() -> None:
    centers = [
        (40.0, 60.0, 8.0),
        (90.0, 80.0, -12.0),
    ]
    spec = _build_gaussian_spectrum(
        shape=(128, 128),
        centers_idx=centers,
        noise_frac=0.005,
        rng_seed=1,
    )

    pos_peaks = detect_peaks(
        spec,
        threshold_rel=0.02,
        min_separation_pts=3,
        refine_centroid=True,
        polarity="positive",
    )
    neg_gt_i = _frac_idx_to_ppm(90.0, spec.indirect_ppm)
    neg_gt_d = _frac_idx_to_ppm(80.0, spec.direct_ppm)

    spacing_ind = abs(float(spec.indirect_ppm[0] - spec.indirect_ppm[1]))
    spacing_dir = abs(float(spec.direct_ppm[0] - spec.direct_ppm[1]))
    tol = 2.0 * max(spacing_ind, spacing_dir)

    near_negative = [
        p
        for p in pos_peaks
        if abs(p.indirect_ppm - neg_gt_i) <= tol and abs(p.direct_ppm - neg_gt_d) <= tol
    ]
    assert not near_negative


def test_corner_peak_detected_refinement_skipped_on_edges() -> None:
    """Corner maximum: still detected; refinement lacks neighbors so ppm stays at pixel centers."""
    n_ind, n_dir = 24, 24
    ind_ppm = np.linspace(50.0, 40.0, n_ind)
    dir_ppm = np.linspace(6.0, 4.0, n_dir)
    z = np.full((n_ind, n_dir), -1.0, dtype=np.float64)
    z[0, 0] = 100.0
    z[0, 1] = 10.0
    z[1, 0] = 10.0

    spec = NMRSpectrum2D(
        data=z,
        direct_ppm=dir_ppm,
        indirect_ppm=ind_ppm,
        direct_nucleus="1H",
        indirect_nucleus="13C",
        metadata={},
    )

    peaks = detect_peaks(
        spec,
        threshold_rel=0.02,
        min_separation_pts=3,
        refine_centroid=True,
        polarity="positive",
    )
    assert peaks
    corner = next(p for p in peaks if p.indirect_idx == 0 and p.direct_idx == 0)
    assert corner.indirect_ppm == float(ind_ppm[0])
    assert corner.direct_ppm == float(dir_ppm[0])


def test_detect_peaks_in_series_order() -> None:
    spec1 = _build_gaussian_spectrum(
        shape=(40, 40),
        centers_idx=[(12.0, 18.0, 5.0)],
        noise_frac=0.01,
        rng_seed=3,
    )
    spec2 = _build_gaussian_spectrum(
        shape=(40, 40),
        centers_idx=[(25.0, 25.0, 6.0)],
        noise_frac=0.01,
        rng_seed=4,
    )

    ts = np.array([0.0, 1.0], dtype=np.float64)
    series = NMRTimeSeries(spectra=[spec1, spec2], timestamps=ts)
    out = detect_peaks_in_series(series, threshold_rel=0.05, min_separation_pts=3)
    assert len(out) == 2
    assert len(out[0]) >= 1 and len(out[1]) >= 1
