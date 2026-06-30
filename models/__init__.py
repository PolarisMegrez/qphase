"""QPhase Model Plugins
---------------------

This directory contains local SDE model plugin definitions for the QPhase
workspace.

Models:
    - Kerr3PA: single-mode Kerr nonlinear cavity with 3-photon absorption
    - Kerr3Mode: three-mode Kerr nonlinear cavity model
    - VDPLevel3: Van der Pol oscillator SDE model
"""

from . import (
    kerr_3mode,
    kerr_3pa,
    vdp_level3,
)

__all__ = [
    "kerr_3pa",
    "kerr_3mode",
    "vdp_level3",
]
