"""
Production System Package
Core Detection Only
"""

__version__ = "1.0.0"
__author__ = "Adversarial Detection Team"
__description__ = "Production-ready adversarial image detection system"

from .detection_core import detect_image_fast, get_decision, get_decision_color

__all__ = [
    'detect_image_fast',
    'get_decision',
    'get_decision_color',
]
