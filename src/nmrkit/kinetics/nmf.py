# SPDX-License-Identifier: Apache-2.0

"""Kinetic component extraction via non-negative matrix factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import NMF

from nmrkit.peaks.track import Trajectory


def _impute_row(row: NDArray[np.float64], mode: Literal["zero", "interp"]) -> NDArray[np.float64]:
    out = np.asarray(row, dtype=np.float64).copy()
    if mode == "zero":
        out = np.where(np.isnan(out), 0.0, out)
        return out
    n = out.shape[0]
    finite = np.flatnonzero(~np.isnan(out))
    if finite.size == 0:
        return np.zeros(n, dtype=np.float64)
    if finite.size == n:
        return out
    idx = np.arange(n, dtype=np.float64)
    xp = finite.astype(np.float64)
    fp = out[finite]
    return np.interp(idx, xp, fp)


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
    # sum
    denom = np.nansum(np.abs(out), axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-15)
    return cast(NDArray[np.float64], out / denom)


def _non_negative_shift(m: NDArray[np.float64]) -> NDArray[np.float64]:
    out = m.astype(np.float64, copy=True)
    row_mins = np.min(out, axis=1, keepdims=True)
    out -= row_mins
    return out


def _run_nmf(
    m_nonneg: NDArray[np.float64],
    n_components: int,
    *,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="cd",
        random_state=random_state,
        max_iter=5000,
        tol=1e-4,
    )
    w = model.fit_transform(m_nonneg)
    h = model.components_
    err = float(model.reconstruction_err_)
    return w.astype(np.float64), h.astype(np.float64), err


def _auto_select_k(ks: list[int], errors: list[float]) -> int:
    """Choose ``k`` by penalized reconstruction error (parsimony over rank).

    Pure reconstruction error keeps decreasing with ``k`` on noisy data; adding a
    linear penalty in ``k`` rewards low-rank solutions that still fit well.
    """
    if len(ks) == 1:
        return ks[0]
    e = np.asarray(errors, dtype=np.float64)
    k_arr = np.asarray(ks, dtype=np.float64)
    k0 = float(k_arr[0])
    # Scale penalty with baseline error at the smallest candidate rank.
    penalty_weight = 0.25 * float(e[0] + 1e-12)
    scores = e + penalty_weight * (k_arr - k0)
    return ks[int(np.argmin(scores))]


@dataclass(frozen=True)
class KineticComponents:
    """NMF decomposition: M ≈ W @ H, where W is peaks×k and H is k×time."""

    n_components: int
    W: np.ndarray
    H: np.ndarray
    trajectory_ids: np.ndarray
    cluster_assignments: np.ndarray


def fit_kinetic_components(
    trajectories: list[Trajectory],
    n_components: int | Literal["auto"] = "auto",
    n_components_max: int = 8,
    normalize: Literal["max", "sum", "none"] = "max",
    impute_missing: Literal["zero", "interp"] = "interp",
    random_state: int = 0,
) -> KineticComponents:
    """Factor intensity curves into kinetic components with sklearn NMF.

    When ``n_components="auto"``, candidate ranks ``k = 2 … min(n_components_max, …)``
    are evaluated and the rank minimizing ``reconstruction_err + λ·(k - k_min)``
    is chosen (``λ`` scales with the baseline error at ``k_min``), discouraging
    excess factors that mostly fit noise.
    """
    if not trajectories:
        msg = "At least one trajectory is required."
        raise ValueError(msg)

    n_peaks = len(trajectories)
    lengths = [int(tr.intensity_curve.shape[0]) for tr in trajectories]
    n_time = max(lengths)

    m_raw = np.full((n_peaks, n_time), np.nan, dtype=np.float64)
    traj_ids = np.empty(n_peaks, dtype=np.int_)
    for i, tr in enumerate(trajectories):
        curve = np.asarray(tr.intensity_curve, dtype=np.float64).ravel()
        if curve.shape[0] != n_time:
            msg = "All intensity_curve arrays must have the same length for stacking."
            raise ValueError(msg)
        m_raw[i, :] = curve
        traj_ids[i] = int(tr.trajectory_id)

    m_imp = np.asarray(
        [_impute_row(m_raw[r], impute_missing) for r in range(n_peaks)],
        dtype=np.float64,
    )
    m_norm = _normalize_rows(m_imp, normalize)
    m_nonneg = _non_negative_shift(m_norm)

    if np.any(np.isnan(m_nonneg)):
        msg = "Non-finite values remain after imputation."
        raise ValueError(msg)

    k_cap = min(n_components_max, n_peaks, n_time)

    if isinstance(n_components, int):
        k = int(n_components)
        if k < 1:
            msg = "n_components must be >= 1."
            raise ValueError(msg)
        if k > k_cap:
            msg = f"n_components={k} exceeds usable rank cap ({k_cap})."
            raise ValueError(msg)
    elif k_cap < 2:
        k = 1
    else:
        ks: list[int] = []
        errs: list[float] = []
        for k_try in range(2, k_cap + 1):
            _w, _h, err = _run_nmf(m_nonneg, k_try, random_state=random_state)
            ks.append(k_try)
            errs.append(err)
        k = _auto_select_k(ks, errs) if ks else 1

    w, h, _err = _run_nmf(m_nonneg, k, random_state=random_state)
    assign = np.argmax(w, axis=1).astype(np.int_)

    return KineticComponents(
        n_components=k,
        W=w,
        H=h,
        trajectory_ids=traj_ids,
        cluster_assignments=assign,
    )
