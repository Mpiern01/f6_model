"""
Safeguards Module
Prevents catastrophic loss and model drift

MIT-level engineering: Comprehensive safety mechanisms
"""

from .catastrophic_loss import CatastrophicLossPrevention
from .anchor_regression import AnchorRegression
from .drift_control import DriftControl
from .gradient_safety import GradientSafety

__all__ = [
    "CatastrophicLossPrevention",
    "AnchorRegression",
    "DriftControl",
    "GradientSafety",
]

