"""
PRODUCTION DETECTION CORE
Heuristic-only version (CNN optional in production, requires GPU)
Uses renamed technical method names

This is the fast path for production batch processing
"""

import numpy as np
from scipy import stats


class HeuristicDetector:
    """Production heuristic detector with technical method names"""

    METHODS = [
        'color_consistency',      # Color balance analysis
        'texture_variance',       # Texture/noise analysis
        'distribution_skew',      # Pixel distribution analysis
        'entropy_pattern',        # Entropy-based fingerprint
        'chromatic_saturation',   # Saturation check
    ]

    @staticmethod
    def detect(image_array):
        """
        Explainable 5-method heuristic detection
        Returns risk_score, confidence, scores dict
        """
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        scores = {}

        # METHOD 1: COLOR CONSISTENCY
        r_mean = np.mean(image_array[:, :, 0])
        g_mean = np.mean(image_array[:, :, 1])
        b_mean = np.mean(image_array[:, :, 2])
        color_std = np.std([r_mean, g_mean, b_mean])
        scores['color_consistency'] = min(max(0, (color_std - 0.15) * 2.0), 1.0)

        # METHOD 2: TEXTURE VARIANCE
        gray = np.mean(image_array, axis=2)
        texture_variance = np.std(gray)
        scores['texture_variance'] = min(max(0, (texture_variance - 0.25) * 2.0), 1.0)

        # METHOD 3: DISTRIBUTION SKEW
        flat = image_array.flatten()
        skewness = abs(stats.skew(flat))
        if np.isnan(skewness):
            skewness = 0.0
        scores['distribution_skew'] = min(max(0, (skewness - 0.2) * 2.0), 1.0)

        # METHOD 4: ENTROPY PATTERN
        hist, _ = np.histogram(gray, bins=256)
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        scores['entropy_pattern'] = min(max(0, (entropy - 5.0) / 4.0), 1.0)

        # METHOD 5: CHROMATIC SATURATION
        if max(r_mean, g_mean, b_mean) > 1e-8:
            saturation = 1.0 - (min(r_mean, g_mean, b_mean) / (max(r_mean, g_mean, b_mean) + 1e-8))
        else:
            saturation = 0.0
        scores['chromatic_saturation'] = (
            1.0 if abs(saturation - 0.5) > 0.35
            else (0.6 if abs(saturation - 0.5) > 0.25 else 0.1)
        )

        # COMBINE SCORES
        risk_score = (
            0.25 * scores['color_consistency'] +
            0.20 * scores['texture_variance'] +
            0.25 * scores['distribution_skew'] +      # Increased - strong indicator
            0.20 * scores['entropy_pattern'] +
            0.10 * scores['chromatic_saturation']
        ) * 100.0

        confidence = max(0.0, min(1.0, 1.0 - (risk_score / 100.0)))

        return risk_score, confidence, scores


def detect_image_fast(image_array):
    """
    Fast heuristic detection (backward compatible)
    Returns: risk_score, confidence, detector_scores
    """
    return HeuristicDetector.detect(image_array)


def get_decision(risk_score):
    """Convert risk score to decision"""
    if risk_score < 50:
        return 'ALLOW'
    elif risk_score < 75:
        return 'REVIEW'
    else:
        return 'BLOCK'


def get_decision_color(decision):
    """Get color for decision"""
    colors = {
        'ALLOW': (0, 255, 0),
        'REVIEW': (255, 165, 0),
        'BLOCK': (255, 0, 0)
    }
    return colors.get(decision, (128, 128, 128))
