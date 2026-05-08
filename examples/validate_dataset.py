# SPDX-License-Identifier: Apache-2.0
"""
End-to-end validation of the nmrkit pipeline on a public dataset.

Usage:
    uv run python examples/validate_dataset.py \\
        --data-dir /path/to/dataset \\
        --output-dir ./validation_output \\
        [--dosy-path /path/to/dosy] \\
        [--n-components 3]

Pipeline:
    1. Load time-resolved HSQC stack
    2. Detect peaks per frame
    3. Track peaks across frames
    4. Fit NMF kinetic components
    5. (Optional) Load DOSY, fit D per peak, refine clusters
    6. Generate summary figure + JSON report

Outputs:
    - summary.png: multi-panel figure
    - clusters.json: trajectory_id → cluster_id, plus mean (δ_H, δ_C, D) per cluster
    - report.md: human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from nmrkit.dosy import fit_diffusion_for_peaks, load_dosy
from nmrkit.io.nmr import load_timeseries
from nmrkit.kinetics import refine_clusters_with_dosy
from nmrkit.kinetics.nmf import KineticComponents, fit_kinetic_components
from nmrkit.peaks import detect_peaks_in_series, track_peaks
from nmrkit.peaks.detect import Peak2D
from nmrkit.peaks.track import Trajectory
from nmrkit.viz import plot_summary


def _impute_row(row: NDArray[np.float64], mode: Literal["zero", "interp"]) -> NDArray[np.float64]:
    out = np.asarray(row, dtype=np.float64).copy()
    if mode == "zero":
        return cast(NDArray[np.float64], np.where(np.isnan(out), 0.0, out))
    n = out.shape[0]
    finite = np.flatnonzero(~np.isnan(out))
    if finite.size == 0:
        return np.zeros(n, dtype=np.float64)
    if finite.size == n:
        return out
    idx = np.arange(n, dtype=np.float64)
    xp = finite.astype(np.float64)
    fp = out[finite]
    return cast(NDArray[np.float64], np.interp(idx, xp, fp))


def _normalize_rows(
    m: NDArray[np.float64],
    mode: Literal["max", "sum", "none"],
) -> NDArray[np.float64]:
    if mode == "none":
        return m.astype(np.float64, copy=False)
    out = m.astype(np.float64, copy=True)
    if mode == "max":
        denom = np.nanmax(np.abs(out), axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-15)
        return cast(NDArray[np.float64], out / denom)
    denom = np.nansum(np.abs(out), axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-15)
    return cast(NDArray[np.float64], out / denom)


def _non_negative_shift(m: NDArray[np.float64]) -> NDArray[np.float64]:
    out = m.astype(np.float64, copy=True)
    row_mins = np.min(out, axis=1, keepdims=True)
    out -= row_mins
    return out


def _nn_input_matrix(
    trajectories: list[Trajectory],
    *,
    normalize: Literal["max", "sum", "none"],
    impute: Literal["zero", "interp"],
) -> NDArray[np.float64]:
    """Match :func:`fit_kinetic_components` stacking for error computation."""
    if not trajectories:
        msg = "No trajectories."
        raise ValueError(msg)
    lengths = [int(tr.intensity_curve.shape[0]) for tr in trajectories]
    n_time = max(lengths)
    m_raw = np.full((len(trajectories), n_time), np.nan, dtype=np.float64)
    for i, tr in enumerate(trajectories):
        curve = np.asarray(tr.intensity_curve, dtype=np.float64).ravel()
        m_raw[i, :] = curve
    m_imp = np.asarray(
        [_impute_row(m_raw[r], impute) for r in range(len(trajectories))],
        dtype=np.float64,
    )
    m_norm = _normalize_rows(m_imp, normalize)
    return _non_negative_shift(m_norm)


def _reconstruction_frobenius(
    components: KineticComponents,
    trajectories: list[Trajectory],
) -> float:
    m = _nn_input_matrix(trajectories, normalize="max", impute="interp")
    wh = np.asarray(components.W @ components.H, dtype=np.float64)
    return float(np.linalg.norm(m - wh, ord="fro"))


def _mean_cluster_coords(
    trajectories: list[Trajectory],
    cluster_assignments: np.ndarray,
    diffusion_by_tid: dict[int, tuple[float, float]] | None,
) -> list[dict[str, object]]:
    by_c: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_assignments.tolist()):
        by_c[int(c)].append(i)

    out: list[dict[str, object]] = []
    for cid in sorted(by_c.keys()):
        idxs = by_c[cid]
        dh = float(np.mean([trajectories[i].direct_ppm_mean for i in idxs]))
        dc = float(np.mean([trajectories[i].indirect_ppm_mean for i in idxs]))
        d_mean: float | None = None
        if diffusion_by_tid:
            vals = []
            for i in idxs:
                tid = int(trajectories[i].trajectory_id)
                if tid in diffusion_by_tid:
                    vals.append(diffusion_by_tid[tid][0])
            if vals:
                d_mean = float(np.mean(vals))
        out.append(
            {
                "cluster_id": cid,
                "n_trajectories": len(idxs),
                "mean_direct_ppm": dh,
                "mean_indirect_ppm": dc,
                "mean_D_m2_s": d_mean,
            }
        )
    return out


def run_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    dosy_path: Path | None,
    n_components: int | str,
    log10_d_tolerance: float,
) -> None:
    print(f"[1/6] Loading time series from {data_dir} …")
    series = load_timeseries(data_dir)
    n_frames = len(series.spectra)
    print(f"      Loaded {n_frames} spectra, timestamps (s): {series.timestamps}")

    print("[2/6] Detecting peaks per frame …")
    peaks_per_frame = detect_peaks_in_series(
        series,
        threshold_rel=0.05,
        min_separation_pts=3,
    )
    counts = [len(p) for p in peaks_per_frame]
    print(f"      Peaks per frame: {counts} (total detections {sum(counts)})")

    print("[3/6] Tracking peaks …")
    trajectories = track_peaks(peaks_per_frame, allow_gap_frames=1)
    print(f"      Trajectories: {len(trajectories)}")

    if not trajectories:
        msg = "No trajectories; cannot fit kinetics. Lower threshold or check data."
        raise RuntimeError(msg)

    print("[4/6] Fitting NMF kinetic components …")
    n_comp_arg: int | Literal["auto"]
    if isinstance(n_components, str) and n_components == "auto":
        n_comp_arg = "auto"
    else:
        n_comp_arg = int(n_components)

    kin = fit_kinetic_components(
        trajectories,
        n_components=n_comp_arg,
        n_components_max=8,
        normalize="max",
        impute_missing="interp",
        random_state=0,
    )
    err = _reconstruction_frobenius(kin, trajectories)
    print(f"      Components: {kin.n_components}, Frobenius ‖M−WH‖_F = {err:.6g}")

    sizes_kinetic: dict[int, int] = {}
    for c in kin.cluster_assignments.tolist():
        sizes_kinetic[int(c)] = sizes_kinetic.get(int(c), 0) + 1
    print(f"      Kinetic cluster sizes: {dict(sorted(sizes_kinetic.items()))}")

    cluster_assign = np.asarray(kin.cluster_assignments, dtype=np.int_).copy()
    diffusion_by_tid: dict[int, tuple[float, float]] | None = None

    if dosy_path is not None:
        print(f"[5/6] Optional DOSY from {dosy_path} …")
        exp = load_dosy(Path(dosy_path))
        peaks_for_d = [tr.peaks[-1] if tr.peaks else _fallback_peak(tr) for tr in trajectories]
        dmap_idx = fit_diffusion_for_peaks(exp, peaks_for_d, integration_window_ppm=0.05)
        diffusion_by_tid = {
            int(trajectories[i].trajectory_id): dmap_idx[i] for i in range(len(trajectories))
        }
        cluster_assign = refine_clusters_with_dosy(
            kin,
            trajectories,
            diffusion_by_tid,
            log10_d_tolerance=log10_d_tolerance,
        )
        print("      Refined cluster assignments with DOSY.")
    else:
        print("[5/6] Skipping DOSY (no --dosy-path).")

    sizes_final: dict[int, int] = {}
    for c in cluster_assign.tolist():
        sizes_final[int(c)] = sizes_final.get(int(c), 0) + 1
    print(f"      Final cluster sizes: {dict(sorted(sizes_final.items()))}")

    print("[6/6] Writing outputs …")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(series, trajectories, kin, output_dir, cluster_assignments=cluster_assign)

    traj_map = {
        str(int(tr.trajectory_id)): int(cluster_assign[i]) for i, tr in enumerate(trajectories)
    }
    clusters = _mean_cluster_coords(trajectories, cluster_assign, diffusion_by_tid)

    report_obj = {
        "trajectory_cluster_map": traj_map,
        "clusters": clusters,
        "n_components": kin.n_components,
        "nmf_frobenius_error": err,
        "dosy_used": dosy_path is not None,
        "cluster_sizes_final": {str(k): v for k, v in sorted(sizes_final.items())},
    }
    with (output_dir / "clusters.json").open("w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2)

    lines = [
        "# nmrkit validation report",
        "",
        f"- Spectra (frames): {n_frames}",
        f"- Trajectories: {len(trajectories)}",
        f"- NMF rank: {kin.n_components}",
        f"- Frobenius reconstruction error: {err:.6g}",
        f"- DOSY refinement: {'yes' if dosy_path else 'no'}",
        "",
        "## Cluster sizes",
        "",
    ]
    for cid, sz in sorted(sizes_final.items()):
        lines.append(f"- Cluster {cid}: {sz} trajectory/trajectories")
    lines.append("")
    lines.append("## Mean coordinates per reported cluster")
    lines.append("")
    for c in clusters:
        d_part = c["mean_D_m2_s"]
        d_str = f"{d_part:.3e}" if isinstance(d_part, float) else "—"
        lines.append(
            f"- Cluster {c['cluster_id']}: δ_H={c['mean_direct_ppm']:.3f} ppm, "
            f"δ_C={c['mean_indirect_ppm']:.3f} ppm, "
            f"⟨D⟩={d_str} m²/s (n={c['n_trajectories']})"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"      Wrote {output_dir / 'summary.png'}, clusters.json, report.md")


def _fallback_peak(tr: Trajectory) -> Peak2D:
    return Peak2D(
        direct_ppm=float(tr.direct_ppm_mean),
        indirect_ppm=float(tr.indirect_ppm_mean),
        intensity=1.0,
        direct_idx=0,
        indirect_idx=0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end nmrkit pipeline validation.")
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory compatible with load_timeseries.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary.png, clusters.json, report.md.",
    )
    p.add_argument("--dosy-path", type=Path, default=None, help="Optional DOSY experiment (.npz).")
    p.add_argument(
        "--n-components",
        default="auto",
        help='Number of NMF components or "auto" (default).',
    )
    p.add_argument(
        "--log10-d-tolerance",
        type=float,
        default=0.15,
        help="DOSY sub-cluster cutoff on |Δ log10 D| (default 0.15).",
    )
    args = p.parse_args()

    nc_raw = args.n_components
    if isinstance(nc_raw, str) and nc_raw.strip().lower() == "auto":
        n_components: int | str = "auto"
    else:
        try:
            n_components = int(nc_raw)
        except (TypeError, ValueError):
            print(f"ERROR: Invalid --n-components: {nc_raw!r}", file=sys.stderr)
            return 1

    try:
        run_pipeline(
            data_dir=args.data_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            dosy_path=args.dosy_path.resolve() if args.dosy_path else None,
            n_components=n_components,
            log10_d_tolerance=float(args.log10_d_tolerance),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
