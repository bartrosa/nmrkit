# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

_BRUKER_TEMPLATE = (
    Path(ng.__file__).resolve().parent / "fileio" / "tests" / "bruker_test_data" / "1"
)


def _write_bruker_2d_pdata(
    *,
    out_pdata: Path,
    shape: tuple[int, int],
    date_unix: int | None,
) -> None:
    """Materialize a minimal Bruker pdata directory using nmrglue's writer."""
    if not _BRUKER_TEMPLATE.is_dir():
        msg = "Bundled nmrglue Bruker test template not found; install nmrglue dev test data."
        raise RuntimeError(msg)

    src_pdata = _BRUKER_TEMPLATE / "pdata" / "1"
    dic, _ = ng.bruker.read_pdata(str(src_pdata))

    data = np.arange(np.prod(shape), dtype="float64").reshape(shape)
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
    ng.bruker.write_pdata(str(out_pdata), dic2, data, write_procs=True, pdata_folder=False)

    exp_dir = out_pdata.parent.parent
    ng.bruker.write_jcamp(dic2["acqus"], str(exp_dir / "acqus"), overwrite=True)
    ng.bruker.write_jcamp(dic2["acqu2s"], str(exp_dir / "acqu2s"), overwrite=True)


@pytest.fixture
def bruker_2d_pdata_path(tmp_path: Path) -> Path:
    exp = tmp_path / "exp"
    shutil.copytree(_BRUKER_TEMPLATE, exp)
    pdata = exp / "pdata" / "1"
    shutil.rmtree(pdata)
    _write_bruker_2d_pdata(out_pdata=pdata, shape=(4, 8), date_unix=1_200_000_000)
    return pdata


@pytest.fixture
def bruker_timeseries_directory(tmp_path: Path) -> Path:
    root = tmp_path / "series"
    root.mkdir()

    base_dates = (1_000_000_000, 1_000_000_600, 1_000_001_200)
    for idx, du in enumerate(base_dates, start=1):
        spec_root = root / f"spec_{idx:03d}"
        shutil.copytree(_BRUKER_TEMPLATE, spec_root)
        pdata = spec_root / "pdata" / "1"
        shutil.rmtree(pdata)
        _write_bruker_2d_pdata(
            out_pdata=pdata,
            shape=(3, 6),
            date_unix=int(du),
        )

    return root
