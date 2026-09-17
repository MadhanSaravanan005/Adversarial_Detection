"""
HYBRID IMAGE INTEGRITY FIREWALL
Two-Stage Detection: Explainable Heuristics + CNN Verification

Architecture:
  Stage 1 (Fast Path): Heuristic anomaly detection (<50ms)
  Stage 2 (Verification): CNN-based secondary check if uncertain
  Stage 3 (Decision): Policy engine combines both scores

This is the CORE DETECTION ENGINE - everything else uses this.
"""

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 1: EXPLAINABLE HEURISTIC DETECTOR
# ============================================================================

class HeuristicDetector:
    """
    First-stage fast screening using explainable image statistics
    Renamed methods for technical clarity
    """

    METHODS = [
        'color_consistency',      # Was: Robustness
        'texture_variance',       # Was: Frequency
        'distribution_skew',      # Was: Statistical
        'entropy_pattern',        # Was: Fingerprinting
        'chromatic_saturation',   # Was: Metadata
    ]

    @staticmethod
    def detect(image_array):
        """
        Run heuristic screening
        Returns: risk_score (0-100), detector_scores dict, is_certain (bool)

        is_certain = True if risk clearly < 30 or > 75 (skip CNN)
        is_certain = False if risk 30-75 (run CNN verification)
        """
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        scores = {}

        # ================================================================
        # METHOD 1: COLOR CONSISTENCY ANALYSIS
        # ================================================================
        # Adversarial attacks often create extreme color imbalance
        r_mean = np.mean(image_array[:, :, 0])
        g_mean = np.mean(image_array[:, :, 1])
        b_mean = np.mean(image_array[:, :, 2])
        color_std = np.std([r_mean, g_mean, b_mean])

        # Normal: 0.05-0.15, Adversarial: 0.15+
        scores['color_consistency'] = min(max(0, (color_std - 0.15) * 2.0), 1.0)

        # ================================================================
        # METHOD 2: TEXTURE VARIANCE ANALYSIS
        # ================================================================
        # Smooth images look natural, high noise looks adversarial
        gray = np.mean(image_array, axis=2)
        texture_variance = np.std(gray)

        # Normal: 0.15-0.35, Adversarial: 0.35+
        scores['texture_variance'] = min(max(0, (texture_variance - 0.25) * 2.0), 1.0)

        # ================================================================
        # METHOD 3: DISTRIBUTION SKEW ANALYSIS
        # ================================================================
        # Natural images have balanced pixel distributions
        flat = image_array.flatten()
        if np.std(flat) < 1e-6:
            skewness = 0.0
        else:
            skewness = abs(float(stats.skew(flat)))
            if np.isnan(skewness):
                skewness = 0.0

        # Normal: 0-0.3, Adversarial: 0.3+
        scores['distribution_skew'] = min(max(0, (skewness - 0.2) * 2.0), 1.0)

        # ================================================================
        # METHOD 4: ENTROPY PATTERN ANALYSIS
        # ================================================================
        # Adversarial noise increases randomness (entropy)
        hist, _ = np.histogram(gray, bins=256)
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))

        # Normal: 4-6 bits, Adversarial: 6-8 bits
        scores['entropy_pattern'] = min(max(0, (entropy - 5.0) / 4.0), 1.0)

        # ================================================================
        # METHOD 5: CHROMATIC SATURATION HEURISTIC
        # ================================================================
        # Natural saturation ~0.5, extreme saturation looks manipulated
        if max(r_mean, g_mean, b_mean) > 1e-8:
            saturation = 1.0 - (min(r_mean, g_mean, b_mean) / (max(r_mean, g_mean, b_mean) + 1e-8))
        else:
            saturation = 0.0

        scores['chromatic_saturation'] = (
            1.0 if abs(saturation - 0.5) > 0.35
            else (0.6 if abs(saturation - 0.5) > 0.25 else 0.1)
        )

        # ================================================================
        # COMBINE HEURISTIC SCORES
        # ================================================================
        heuristic_risk = (
            0.25 * scores['color_consistency'] +
            0.20 * scores['texture_variance'] +
            0.25 * scores['distribution_skew'] +      # Increased - strong indicator
            0.20 * scores['entropy_pattern'] +
            0.10 * scores['chromatic_saturation']
        ) * 100.0

        # Determine if we need CNN verification
        is_certain = heuristic_risk < 30 or heuristic_risk > 75

        return heuristic_risk, scores, is_certain


# ============================================================================
# STAGE 2: CNN SECONDARY VERIFIER
# ============================================================================

class AdversarialCNNVerifier(nn.Module):
    """
    Secondary CNN stage for uncertain cases
    Only runs when heuristic score is 30-75
    """

    def __init__(self):
        super(AdversarialCNNVerifier, self).__init__()

        # Lightweight CNN designed for 32x32 images
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (16, 16, 16)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (32, 8, 8)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (64, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # Binary: clean (0) vs adversarial (1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_risk(self, image_array):
        """
        Convert image to risk score (0-1)
        """
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(image_array).float()
        tensor = tensor.permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self(tensor)
            probs = torch.softmax(outputs, dim=1)
            # Risk = probability of being adversarial
            cnn_risk = probs[0, 1].item()

        return cnn_risk


# ============================================================================
# STAGE 3: HYBRID POLICY ENGINE
# ============================================================================

class HybridDetector:
    """
    Complete hybrid detection system
    Combines heuristic + CNN + policy engine

    This is what the API actually uses!
    """

    def __init__(self, cnn_model=None, device=None):
        self.heuristic = HeuristicDetector()

        # Initialize CNN if provided
        if cnn_model is None:
            self.cnn = AdversarialCNNVerifier()
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cnn = self.cnn.to(device)
        else:
            self.cnn = cnn_model

        self.device = device or torch.device('cpu')
        self.cnn.eval()  # Evaluation mode

        # Check for pretrained model checkpoint
        from pathlib import Path
        model_path = Path(__file__).resolve().parent.parent / "models" / "robust_model.pth"
        self.has_checkpoint = False
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.cnn.load_state_dict(state_dict)
                self.has_checkpoint = True
                logger.info(f"[HybridDetector] Loaded weights from {model_path}")
            except Exception as e:
                logger.warning(f"[HybridDetector] Could not load checkpoint from {model_path}: {e}")
        else:
            logger.info("[HybridDetector] No pretrained checkpoint found at models/robust_model.pth (operating with initialized architecture).")

        logger.info(f"[HybridDetector] Initialized on {self.device}")
        logger.info(f"[HybridDetector] Heuristic methods: {', '.join(self.heuristic.METHODS)}")

    def detect(self, image_array, use_cnn=True):
        """
        Complete hybrid detection pipeline

        Returns:
          - final_risk_score: 0-100
          - decision: 'ALLOW' | 'REVIEW' | 'BLOCK'
          - explanation: dict with all scores and reasoning
        """

        # ================================================================
        # STAGE 1: Fast heuristic screening
        # ================================================================
        heuristic_risk, heuristic_scores, is_certain = self.heuristic.detect(image_array)

        explanation = {
            'stage': '1_heuristic',
            'heuristic_risk': float(heuristic_risk),
            'heuristic_scores': {k: float(v) for k, v in heuristic_scores.items()},
            'is_certain': is_certain,
            'cnn_used': False,
            'final_risk': float(heuristic_risk),
        }

        # ================================================================
        # STAGE 2: CNN verification for uncertain cases (if checkpoint available)
        # ================================================================
        if use_cnn and not is_certain:
            if getattr(self, 'has_checkpoint', False):
                try:
                    # Apply 3-layer preprocessing defense (JPEG -> Blur -> Median) before CNN evaluation
                    try:
                        from production_system.SANITIZATION_APPROACH import AdversarialImageSanitizer
                        cleaned_image = AdversarialImageSanitizer.apply_pipeline(
                            image_array, jpeg_quality=85, blur_sigma=1.0, median_kernel=3
                        )
                    except Exception as prep_err:
                        logger.warning(f"Preprocessing defense failed ({prep_err}), evaluating on raw image")
                        cleaned_image = image_array

                    raw_cnn = self.cnn.predict_risk(cleaned_image)
                    cnn_risk = float(raw_cnn * 100.0 if raw_cnn <= 1.0 else raw_cnn)
                    cnn_risk = min(max(0.0, cnn_risk), 100.0)

                    # Combine scores: heuristic is more certain, CNN provides second opinion
                    final_risk = min(max(0.0, (0.65 * heuristic_risk + 0.35 * cnn_risk)), 100.0)

                    explanation.update({
                        'stage': '2_hybrid',
                        'cnn_available': True,
                        'cnn_used': True,
                        'cnn_risk': float(cnn_risk),
                        'cnn_weight': 0.35,
                        'heuristic_weight': 0.65,
                        'final_risk': float(final_risk),
                    })

                except Exception as e:
                    logger.warning(f"CNN verification failed: {e}. Using heuristic only.")
                    final_risk = min(max(0.0, heuristic_risk), 100.0)
                    explanation['cnn_error'] = str(e)
            else:
                # Graceful degradation: No trained checkpoint loaded
                logger.info("[HybridDetector] Stage 2 CNN skipped: No trained checkpoint loaded. Relying on Stage 1 heuristic ensemble.")
                final_risk = min(max(0.0, heuristic_risk), 100.0)
                explanation.update({
                    'stage': '1_heuristic_ensemble',
                    'cnn_available': False,
                    'cnn_used': False,
                    'cnn_status': 'checkpoint_not_loaded (operating in heuristic-only mode)',
                    'final_risk': float(final_risk),
                })
        else:
            final_risk = min(max(0.0, heuristic_risk), 100.0)
            explanation.update({
                'cnn_available': getattr(self, 'has_checkpoint', False),
                'cnn_used': False,
            })

        # ================================================================
        # STAGE 3: Policy engine - convert risk to decision
        # ================================================================
        if final_risk < 50:
            decision = 'ALLOW'
            reason = 'Low risk - image appears legitimate'
        elif final_risk < 75:
            decision = 'REVIEW'
            reason = 'Uncertain - manual review recommended'
        else:
            decision = 'BLOCK'
            reason = 'High risk - likely adversarial or suspicious'

        explanation.update({
            'decision': decision,
            'reason': reason,
            'thresholds': {
                'allow_max': 50,
                'review_range': [50, 75],
                'block_min': 75,
            }
        })

        return final_risk, decision, explanation

    def train_step(self, images_batch, labels_batch):
        """
        Train the CNN on detected examples
        Called during retraining phase
        """
        if self.cnn is None:
            return None

        try:
            # Convert to tensors
            images_tensor = torch.from_numpy(images_batch).float()
            images_tensor = images_tensor.permute(0, 3, 1, 2)  # (N,H,W,3) -> (N,3,H,W)
            images_tensor = images_tensor.to(self.device)

            labels_tensor = torch.from_numpy(labels_batch).long().to(self.device)

            # Training
            self.cnn.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.00001)

            optimizer.zero_grad()
            outputs = self.cnn(images_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            self.cnn.eval()

            loss_value = loss.item()
            logger.info(f"[CNN Training] Loss: {loss_value:.6f}")
            return loss_value

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None


# ============================================================================
# HELPER FUNCTIONS (Backward compatible)
# ============================================================================

def detect_image_fast(image_array):
    """
    Fast heuristic detection wrapper (backward compatible)
    Returns: risk_score, confidence, detector_scores
    """
    risk_score, scores, _ = HeuristicDetector.detect(image_array)
    confidence = max(0.0, min(1.0, 1.0 - (risk_score / 100.0)))
    return risk_score, confidence, scores


def get_decision(risk_score):
    """Converts risk to decision"""
    if risk_score < 50:
        return 'ALLOW'
    elif risk_score < 75:
        return 'REVIEW'
    else:
        return 'BLOCK'


def get_decision_color(decision):
    """Legacy function - decision to color"""
    colors = {
        'ALLOW': (0, 255, 0),    # Green
        'REVIEW': (255, 165, 0),  # Orange
        'BLOCK': (255, 0, 0)      # Red
    }
    return colors.get(decision, (128, 128, 128))
