"""
QPhaseSDE: Noise Models Subpackage
---------------------------------
Provides stochastic noise model factories and implementations (e.g., Gaussian),
exposed via a small factory for engine consumption.
"""

from .gaussian import GaussianNoiseModel
from .factory import make_noise_model

__all__ = ["GaussianNoiseModel", "make_noise_model"]
