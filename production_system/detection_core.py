"""
Production Detection Core
Re-exports canonical HeuristicDetector from backend.hybrid_detector.
Eliminates code duplication while preserving backward compatibility.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.hybrid_detector import (
    HeuristicDetector,
    detect_image_fast,
    get_decision,
    get_decision_color
)

__all__ = [
    "HeuristicDetector",
    "detect_image_fast",
    "get_decision",
    "get_decision_color",
]
