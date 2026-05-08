# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import nmrkit


def test_version_is_set() -> None:
    assert nmrkit.__version__
    assert isinstance(nmrkit.__version__, str)
