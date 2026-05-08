# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import nmrglue as ng
import numpy as np
from nmrglue.fileio import fileiobase
from numpy.typing import NDArray

FormatHint = Literal["bruker_pdata", "nmrpipe_ft2"]


def _normalize_nucleus(label: object) -> str:
    s = str(label).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    return s if s else "?"


def _to_real_2d(data: np.ndarray) -> NDArray[np.float64]:
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        return cast(NDArray[np.float64], np.abs(arr).astype(np.float64, copy=False))
    return cast(NDArray[np.float64], arr.astype(np.float64, copy=False))


def _maybe_flip_ppm_axis(ppm: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ppm ordered high → low and whether the spectral axis was reversed."""
    if ppm.size < 2:
        return ppm.astype(np.float64, copy=False), False
    if ppm[0] < ppm[-1]:
        return ppm[::-1].astype(np.float64, copy=False), True
    return ppm.astype(np.float64, copy=False), False


def _align_two_d_axes(
    data: np.ndarray,
    indirect_ppm: np.ndarray,
    direct_ppm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure descending ppm on both axes; flip data slices when reversing."""
    out = data
    ind_ppm, ind_flip = _maybe_flip_ppm_axis(indirect_ppm)
    dir_ppm, dir_flip = _maybe_flip_ppm_axis(direct_ppm)
    if ind_flip:
        out = np.flip(out, axis=0)
    if dir_flip:
        out = np.flip(out, axis=1)
    return out, ind_ppm, dir_ppm


def _resolve_timeseries_entry(path: Path) -> Path | None:
    """Return a concrete `.ft2` path or Bruker pdata directory containing ``2rr``."""
    if path.is_file():
        return path if path.suffix.lower() == ".ft2" else None
    if not path.is_dir():
        return None
    if (path / "2rr").is_file():
        return path
    pdata_root = path / "pdata"
    if pdata_root.is_dir():
        for child in sorted(pdata_root.iterdir()):
            if child.is_dir() and (child / "2rr").is_file():
                return child
    return None


def _detect_format(path: Path) -> FormatHint:
    if path.is_file():
        if path.suffix.lower() == ".ft2":
            return "nmrpipe_ft2"
        msg = f"Unsupported spectrum file (expected .ft2): {path}"
        raise ValueError(msg)
    if path.is_dir():
        if (path / "2rr").is_file():
            return "bruker_pdata"
        msg = f"Directory does not look like Bruker pdata (missing 2rr): {path}"
        raise ValueError(msg)
    msg = f"Spectrum path is not a file or directory: {path}"
    raise ValueError(msg)


def _load_bruker_pdata(path: Path) -> NMRSpectrum2D:
    dic, data = ng.bruker.read_pdata(str(path))
    arr = _to_real_2d(data)
    if arr.ndim != 2:
        msg = f"Expected 2D Bruker pdata, got shape {arr.shape}"
        raise ValueError(msg)

    udic = ng.bruker.guess_udic(dic, arr, strip_fake=True)
    if int(udic["ndim"]) != 2:
        msg = f"Expected 2D universal dictionary, got ndim={udic['ndim']!r}"
        raise ValueError(msg)

    uc_indirect = fileiobase.uc_from_udic(udic, dim=0)
    uc_direct = fileiobase.uc_from_udic(udic, dim=1)
    indirect_ppm_raw = uc_indirect.ppm_scale()
    direct_ppm_raw = uc_direct.ppm_scale()

    arr, indirect_ppm, direct_ppm = _align_two_d_axes(arr, indirect_ppm_raw, direct_ppm_raw)

    meta: dict[str, Any] = {
        "format": "bruker_pdata",
        "path": str(path.resolve()),
        "dic": dic,
    }

    return NMRSpectrum2D(
        data=arr,
        direct_ppm=direct_ppm,
        indirect_ppm=indirect_ppm,
        direct_nucleus=_normalize_nucleus(udic[1]["label"]),
        indirect_nucleus=_normalize_nucleus(udic[0]["label"]),
        metadata=meta,
    )


def _load_nmrpipe_ft2(path: Path) -> NMRSpectrum2D:
    dic, data = ng.pipe.read(str(path))
    arr = _to_real_2d(data)
    if arr.ndim != 2:
        msg = f"Expected 2D NMRPipe spectrum, got shape {arr.shape}"
        raise ValueError(msg)

    udic = ng.pipe.guess_udic(dic, arr)
    if int(udic["ndim"]) != 2:
        msg = f"Expected 2D NMRPipe data, got ndim={udic['ndim']!r}"
        raise ValueError(msg)

    uc_indirect = ng.pipe.make_uc(dic, arr, dim=0)
    uc_direct = ng.pipe.make_uc(dic, arr, dim=-1)
    indirect_ppm_raw = uc_indirect.ppm_scale()
    direct_ppm_raw = uc_direct.ppm_scale()

    arr, indirect_ppm, direct_ppm = _align_two_d_axes(arr, indirect_ppm_raw, direct_ppm_raw)

    meta: dict[str, Any] = {
        "format": "nmrpipe_ft2",
        "path": str(path.resolve()),
        "dic": dic,
    }

    return NMRSpectrum2D(
        data=arr,
        direct_ppm=direct_ppm,
        indirect_ppm=indirect_ppm,
        direct_nucleus=_normalize_nucleus(udic[1]["label"]),
        indirect_nucleus=_normalize_nucleus(udic[0]["label"]),
        metadata=meta,
    )


def _extract_unix_timestamp(metadata: dict[str, Any]) -> float | None:
    """Best-effort acquisition/storage timestamp from loader metadata."""
    fmt = metadata.get("format")
    if fmt == "bruker_pdata":
        dic = metadata.get("dic")
        if isinstance(dic, dict):
            acqus = dic.get("acqus")
            if isinstance(acqus, dict) and "DATE" in acqus:
                return float(acqus["DATE"])
        return None
    if fmt == "nmrpipe_ft2":
        dic = metadata.get("dic")
        if isinstance(dic, dict):
            for key in sorted(dic.keys()):
                if not key.endswith("DATE"):
                    continue
                val = dic[key]
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None
    return None


@dataclass(frozen=True)
class NMRSpectrum2D:
    """A processed 2D NMR spectrum with axis calibration."""

    data: np.ndarray  # shape (n_indirect, n_direct), real-valued, units arbitrary
    direct_ppm: np.ndarray  # 1D, length n_direct, descending (high → low ppm convention)
    indirect_ppm: np.ndarray  # 1D, length n_indirect, descending
    direct_nucleus: str  # e.g. "1H"
    indirect_nucleus: str  # e.g. "13C"
    metadata: dict[str, Any]  # raw metadata for traceability


@dataclass(frozen=True)
class NMRTimeSeries:
    """An ordered series of 2D spectra with timestamps."""

    spectra: list[NMRSpectrum2D]
    timestamps: np.ndarray  # 1D, length len(spectra), seconds since start


def load_spectrum_2d(path: Path) -> NMRSpectrum2D:
    """Load a single processed 2D spectrum (Bruker pdata dir or NMRPipe `.ft2` file)."""
    p = Path(path)
    kind = _detect_format(p)
    if kind == "bruker_pdata":
        return _load_bruker_pdata(p)
    return _load_nmrpipe_ft2(p)


def load_timeseries(
    directory: Path,
    glob_pattern: str = "*",
    time_source: str = "metadata",
) -> NMRTimeSeries:
    """Load an ordered series of 2D spectra from entries matched under ``directory``.

    Each entry may be a NMRPipe ``.ft2`` file, a Bruker pdata directory containing
    ``2rr``, or a Bruker experiment directory whose ``pdata/<n>/`` folder holds ``2rr``.
    """
    if time_source != "metadata":
        msg = f"Unsupported time_source: {time_source!r}"
        raise ValueError(msg)

    root = Path(directory)
    if not root.is_dir():
        msg = f"Not a directory: {root}"
        raise ValueError(msg)

    entries = sorted(root.glob(glob_pattern))
    entries = [p for p in entries if p.name != "." and p.name != ".."]
    entries = [p for p in entries if not p.name.startswith(".")]

    spectra: list[NMRSpectrum2D] = []
    for p in entries:
        target = _resolve_timeseries_entry(p)
        if target is None:
            continue
        spectra.append(load_spectrum_2d(target))

    if not spectra:
        msg = f"No loadable 2D spectra found under {root} with pattern {glob_pattern!r}"
        raise ValueError(msg)

    raw_times = [_extract_unix_timestamp(sp.metadata) for sp in spectra]
    if all(t is not None for t in raw_times):
        times_s = [float(t) for t in raw_times if t is not None]
        t0 = min(times_s)
        stamps = np.array([float(x - t0) for x in times_s], dtype=np.float64)
    else:
        warnings.warn(
            "Could not read timestamps from metadata for all spectra; "
            "using synthetic 0, 1, 2, … seconds.",
            stacklevel=2,
        )
        stamps = np.arange(len(spectra), dtype=np.float64)

    return NMRTimeSeries(spectra=spectra, timestamps=stamps)
