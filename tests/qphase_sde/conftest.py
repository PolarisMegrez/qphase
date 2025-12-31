"""Pytest configuration for qphase_sde tests."""

import sys
from pathlib import Path

# Add qphase_sde to path
# This file is in tests/qphase_sde/
# Root is ../../
packages_dir = Path(__file__).parents[2] / "packages"
sys.path.insert(0, str(packages_dir / "qphase_sde"))
