# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import nmrglue as ng
import numpy as np

from nmrkit.io.nmr import load_spectrum_2d, load_timeseries

_NMRPIPE_2D_FT2 = (
    Path(ng.__file__).resolve().parent / "fileio" / "tests" / "data" / "nmrpipe_2d_freq.ft2"
)


def test_load_nmrpipe_2d_ft2():
    assert _NMRPIPE_2D_FT2.is_file()
    spec = load_spectrum_2d(_NMRPIPE_2D_FT2)
    assert spec.data.ndim == 2
    assert spec.direct_ppm[0] >= spec.direct_ppm[-1]
    assert spec.indirect_ppm[0] >= spec.indirect_ppm[-1]
    assert spec.metadata.get("format") == "nmrpipe_ft2"


def test_load_bruker_2d_roundtrip(bruker_2d_pdata_path):
    spec = load_spectrum_2d(bruker_2d_pdata_path)
    assert spec.data.shape == (4, 8)
    assert spec.direct_ppm.shape == (8,)
    assert spec.indirect_ppm.shape == (4,)
    assert spec.direct_ppm[0] >= spec.direct_ppm[-1]
    assert spec.indirect_ppm[0] >= spec.indirect_ppm[-1]
    assert spec.direct_nucleus
    assert spec.indirect_nucleus
    assert spec.metadata.get("format") == "bruker_pdata"


def test_load_timeseries_order_and_times(bruker_timeseries_directory):
    ts = load_timeseries(bruker_timeseries_directory, glob_pattern="*")
    assert len(ts.spectra) == 3
    assert ts.timestamps.shape == (3,)
    assert np.all(np.diff(ts.timestamps) > 0)
    assert ts.timestamps[0] == 0.0
