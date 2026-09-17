"""
INTEGRATED HYBRID DETECTION WITH SANITIZATION
Complete system: Detect → Try to Recover → Final Decision

This module combines:
1. Hybrid detection (heuristic + CNN)
2. Automatic sanitization attempt if risky
3. Recovery report (success/failure)
"""

import numpy as np
import torch
import logging
import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from production_system.SANITIZATION_APPROACH import AdversarialImageSanitizer
except ImportError:
    try:
        from SANITIZATION_APPROACH import AdversarialImageSanitizer
    except ImportError:
        from ..production_system.SANITIZATION_APPROACH import AdversarialImageSanitizer

logger = logging.getLogger(__name__)


class HybridDetectionWithRecovery:
    """
    Complete detection pipeline with automatic recovery attempt

    Workflow:
    1. Detect image (heuristic + CNN)
    2. If risky (40-100%), try to recover using sanitization
    3. Re-detect recovered image
    4. Final decision: use best outcome
    """

    def __init__(self, hybrid_detector=None, device=None):
        self.detector = hybrid_detector
        self.device = device or torch.device('cpu')

        # Defense techniques to try in order
        self.defenses = [
            ("JPEG_Q90", lambda x: AdversarialImageSanitizer.jpeg_compression(x, quality=90)),
            ("JPEG_Q75", lambda x: AdversarialImageSanitizer.jpeg_compression(x, quality=75)),
            ("Resize_50", lambda x: AdversarialImageSanitizer.resizing_defense(x, downscale=0.5)),
            ("Blur_2.0", lambda x: AdversarialImageSanitizer.gaussian_blur(x, sigma=2.0)),
            ("Median_5", lambda x: AdversarialImageSanitizer.median_filter(x, kernel_size=5)),
            ("BitDepth_4", lambda x: AdversarialImageSanitizer.bit_depth_reduction(x, bits=4)),
        ]

        logger.info("[HybridDetectionWithRecovery] Initialized with {} defense techniques".format(len(self.defenses)))

    def detect_with_recovery(self, image_array):
        """
        Complete detection pipeline with recovery attempt

        Returns:
          - original_risk: Initial risk score
          - final_risk: Risk after recovery (if attempted)
          - final_decision: ALLOW | REVIEW | BLOCK
          - recovery_info: Details about recovery attempt
          - explanation: Full decision explanation
        """

        # Stage 1: Initial Detection
        logger.info("[Pipeline] Stage 1: Initial detection...")
        original_risk, original_decision, original_explanation = self.detector.detect(
            image_array,
            use_cnn=True
        )

        recovery_info = {
            'attempted': False,
            'success': False,
            'technique_used': None,
            'recovered_risk': None,
            'recovered_image': None,
        }

        # Stage 2: Decide if Recovery Needed
        if original_risk < 50:
            # Image is safe - no recovery needed
            logger.info(f"[Pipeline] Image safe ({original_risk:.1f}%). No recovery needed.")
            return {
                'original_risk': float(original_risk),
                'final_risk': float(original_risk),
                'final_decision': original_decision,
                'recovery_attempted': False,
                'recovery_successful': False,
                'explanation': original_explanation,
                'recovery_info': recovery_info
            }

        # Image is risky - try to recover
        logger.info(f"[Pipeline] Image risky ({original_risk:.1f}%). Attempting recovery...")
        recovery_info['attempted'] = True

        # Stage 3: Try Each Defense
        best_recovered_risk = original_risk
        best_recovered_image = None
        best_technique = None

        for technique_name, defense_func in self.defenses:
            try:
                # Apply defense
                recovered = defense_func(image_array)

                # Re-detect
                recovered_risk, recovered_decision, _ = self.detector.detect(
                    recovered,
                    use_cnn=True
                )

                logger.info(
                    f"  [{technique_name}] Original: {original_risk:.1f}% → "
                    f"After: {recovered_risk:.1f}%"
                )

                # Check if this is better
                if recovered_risk < best_recovered_risk:
                    best_recovered_risk = recovered_risk
                    best_recovered_image = recovered
                    best_technique = technique_name

                    # Early exit if we achieved ALLOW
                    if recovered_risk < 50:
                        logger.info(
                            f"  [SUCCESS] {technique_name} recovered image! "
                            f"Risk: {best_recovered_risk:.1f}%"
                        )
                        break

            except Exception as e:
                logger.warning(f"  [{technique_name}] Failed: {str(e)}")
                continue

        # Stage 4: Final Decision
        if best_recovered_image is not None:
            recovery_info['success'] = best_recovered_risk < 50
            recovery_info['technique_used'] = best_technique
            recovery_info['recovered_risk'] = float(best_recovered_risk)
            recovery_info['recovered_image'] = best_recovered_image
            recovery_info['improvement'] = float(original_risk - best_recovered_risk)

            final_risk = best_recovered_risk
            final_decision = 'ALLOW' if best_recovered_risk < 50 else (
                'REVIEW' if best_recovered_risk < 75 else 'BLOCK'
            )

            if best_recovered_risk < 50:
                result_message = "RECOVERY SUCCESSFUL - Image recovered"
            else:
                result_message = f"PARTIAL RECOVERY - Risk reduced by {original_risk - best_recovered_risk:.1f}%"

            logger.info(f"[Pipeline] Stage 4 Decision: {result_message}")
        else:
            final_risk = original_risk
            final_decision = original_decision
            logger.info("[Pipeline] No recovery technique improved the result")

        # Build final explanation
        final_explanation = original_explanation.copy()
        final_explanation.update({
            'recovery_attempted': recovery_info['attempted'],
            'recovery_successful': recovery_info['success'],
            'recovery_technique': recovery_info['technique_used'],
            'original_risk': float(original_risk),
            'final_risk': float(final_risk),
            'risk_improvement': float(original_risk - final_risk) if final_risk < original_risk else 0,
        })

        return {
            'original_risk': float(original_risk),
            'final_risk': float(final_risk),
            'final_decision': final_decision,
            'recovery_attempted': recovery_info['attempted'],
            'recovery_successful': recovery_info['success'],
            'explanation': final_explanation,
            'recovery_info': recovery_info
        }


# ============================================================================
# WRAPPER FUNCTION FOR API
# ============================================================================

def detect_with_automatic_recovery(image_array, detector, use_cnn=True):
    """
    Simple wrapper for API integration

    Returns:
      - final_risk: Risk score [0-100]
      - final_decision: ALLOW | REVIEW | BLOCK
      - recovery_used: bool (was recovery attempted)
      - details: Full recovery details
    """

    pipeline = HybridDetectionWithRecovery(detector)
    result = pipeline.detect_with_recovery(image_array)

    return (
        result['final_risk'],
        result['final_decision'],
        result['recovery_attempted'] and result['recovery_successful'],
        {
            'original_risk': result['original_risk'],
            'recovery_info': result['recovery_info'],
            'explanation': result['explanation']
        }
    )
