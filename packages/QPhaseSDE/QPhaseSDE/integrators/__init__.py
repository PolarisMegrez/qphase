"""
QPhaseSDE: Integrators Subpackage
--------------------------------
Time-stepping schemes for stochastic differential equations (SDEs), wired to
the central registry. Currently provides Euler–Maruyama (with aliases); more
methods may be added in future releases.
"""

from .euler_maruyama import EulerMaruyama

__all__ = ["EulerMaruyama"]
