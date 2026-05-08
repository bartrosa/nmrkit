# SPDX-License-Identifier: Apache-2.0
"""Materialize ``tests/fixtures/synthetic_timeseries/`` for CI and local demos.

Run from repo root:
    uv run python tests/fixtures/build_synthetic_timeseries.py
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import nmrglue as ng
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_ROOT = _REPO_ROOT / "tests" / "fixtures" / "synthetic_timeseries"

_BRUKER_TEMPLATE = (
    Path(ng.__file__).resolve().parent / "fileio" / "tests" / "bruker_test_data" / "1"
)


def _write_bruker_2d_pdata(
    *,
    out_pdata: Path,
    shape: tuple[int, int],
    date_unix: int | None,
    data: np.ndarray | None = None,
) -> None:
    if not _BRUKER_TEMPLATE.is_dir():
        msg = "nmrglue Bruker test template not found."
        raise RuntimeError(msg)

    src_pdata = _BRUKER_TEMPLATE / "pdata" / "1"
    dic, _ = ng.bruker.read_pdata(str(src_pdata))

    if data is None:
        arr = np.arange(np.prod(shape), dtype="float64").reshape(shape)
    else:
        arr = np.asarray(data, dtype="float64")
        if arr.shape != shape:
            msg = f"data shape {arr.shape} != {shape}"
            raise ValueError(msg)

    dic2 = copy.deepcopy(dic)

    dic2["proc2s"] = copy.deepcopy(dic["procs"])
    dic2["procs"]["SI"] = shape[1]
    dic2["procs"]["FTSIZE"] = shape[1]
    dic2["procs"]["XDIM"] = shape[1]

    dic2["proc2s"]["SI"] = shape[0]
    dic2["proc2s"]["FTSIZE"] = shape[0]
    dic2["proc2s"]["XDIM"] = shape[0]
    dic2["proc2s"]["AXNUC"] = "<13C>"
    dic2["proc2s"]["SW_p"] = dic2["proc2s"].get("SW_p", 10.0)

    dic2["acqu2s"] = copy.deepcopy(dic["acqus"])
    dic2["acqu2s"]["NUC1"] = "<13C>"
    dic2["acqu2s"]["TD"] = shape[0]

    if date_unix is not None:
        dic2["acqus"]["DATE"] = int(date_unix)

    out_pdata.parent.mkdir(parents=True, exist_ok=True)
    ng.bruker.write_pdata(str(out_pdata), dic2, arr, write_procs=True, pdata_folder=False)

    exp_dir = out_pdata.parent.parent
    ng.bruker.write_jcamp(dic2["acqus"], str(exp_dir / "acqus"), overwrite=True)
    ng.bruker.write_jcamp(dic2["acqu2s"], str(exp_dir / "acqu2s"), overwrite=True)


def _synthetic_surface(shape: tuple[int, int], *, amp1: float, amp2: float) -> np.ndarray:
    ny, nx = shape
    ii, jj = np.indices((ny, nx))
    p1 = np.exp(-((ii - 2.0) ** 2 + (jj - 4.0) ** 2) / (2 * 1.4**2))
    p2 = np.exp(-((ii - 5.5) ** 2 + (jj - 11.0) ** 2) / (2 * 1.4**2))
    return 0.02 + 8.0 * amp1 * p1 + 8.0 * amp2 * p2


def main() -> None:
    if not _BRUKER_TEMPLATE.is_dir():
        print("ERROR: nmrglue Bruker template missing.", file=sys.stderr)
        sys.exit(1)

    shape = (8, 16)
    base_dates = (1_000_000_000, 1_000_000_600, 1_000_001_200)
    amps = ((1.0, 0.15), (0.55, 0.55), (0.15, 1.0))

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for existing in _OUT_ROOT.iterdir():
        if existing.name.startswith("."):
            continue
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    for idx, du in enumerate(base_dates, start=1):
        spec_root = _OUT_ROOT / f"spec_{idx:03d}"
        shutil.copytree(_BRUKER_TEMPLATE, spec_root)
        pdata = spec_root / "pdata" / "1"
        shutil.rmtree(pdata)
        a1, a2 = amps[idx - 1]
        surface = _synthetic_surface(shape, amp1=a1, amp2=a2)
        _write_bruker_2d_pdata(
            out_pdata=pdata,
            shape=shape,
            date_unix=int(du),
            data=surface,
        )

    print(f"Wrote three experiments under {_OUT_ROOT}")


if __name__ == "__main__":
    main()
