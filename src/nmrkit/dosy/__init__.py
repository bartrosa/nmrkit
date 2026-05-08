# SPDX-License-Identifier: Apache-2.0

"""DOSY diffusion analysis."""

from nmrkit.dosy.fit import (
    DOSYExperiment,
    fit_diffusion_coefficient,
    fit_diffusion_for_peaks,
    load_dosy,
)

__all__ = [
    "DOSYExperiment",
    "fit_diffusion_coefficient",
    "fit_diffusion_for_peaks",
    "load_dosy",
]
