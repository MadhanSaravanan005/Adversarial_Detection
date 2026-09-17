"""
Production-Ready Flask Backend
AI-Generated Media Detection Firewall
Clean, Fast, No Unicode Issues
"""

import os
import logging
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
from scipy import stats
import torch
from backend.background_trainer import BackgroundTrainer, AdversarialImageBuffer
from backend.hybrid_detector import HybridDetector
from backend.hybrid_detection_with_recovery import HybridDetectionWithRecovery

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = str(PROJECT_ROOT / 'uploads')
LOGS_FOLDER = str(PROJECT_ROOT / 'logs')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Global state - Real metrics tracking
class SystemState:
    def __init__(self):
        self.buffer_size = 0
        self.buffer_max = 5
        self.retrain_count = 0
        self.baseline_accuracy = None  # No pre-trained benchmark checkpoint loaded
        self.current_accuracy = None
        self.total_detections = 0
        self.allow_count = 0
        self.review_count = 0
        self.block_count = 0

state = SystemState()

# ============================================================================
# BACKGROUND REAL-TIME RETRAINING SYSTEM
# ============================================================================

# Initialize adversarial buffer for background retraining
adv_buffer = AdversarialImageBuffer(max_size=100)

# Initialize CNN model for hybrid detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from backend.background_trainer import AdversarialDetectionCNN
cnn_model = AdversarialDetectionCNN()
hybrid_detector = HybridDetector(cnn_model=cnn_model, device=device)

# Initialize background trainer thread
background_trainer = BackgroundTrainer(
    model=cnn_model,
    device=device,
    buffer=adv_buffer,
    buffer_threshold=5
)
background_trainer.start()
logger.info("[OK] Hybrid Detection System initialized (Heuristic + CNN)")
# DETECTION ENGINE (5-Method Ensemble)
# ============================================================================

# ============================================================================
# DETECTION ENGINE: Use Hybrid Detector
# ============================================================================

def detect_image_fast(image_array):
    """
    Wrapper function maintaining backward compatibility
    Calls the hybrid detector (heuristic + CNN)
    """
    # Use hybrid detector
    risk_score, decision, explanation = hybrid_detector.detect(image_array, use_cnn=True)

    # Extract detector_scores from explanation for compatibility
    detector_scores = explanation.get('heuristic_scores', {})
    # Rename keys for backward compatibility
    if 'color_consistency' in detector_scores:
        detector_scores = {
            'robustness': detector_scores.get('color_consistency', 0),
            'frequency': detector_scores.get('texture_variance', 0),
            'statistical': detector_scores.get('distribution_skew', 0),
            'fingerprinting': detector_scores.get('entropy_pattern', 0),
            'metadata': detector_scores.get('chromatic_saturation', 0),
        }

    confidence = max(0.0, min(1.0, 1.0 - (risk_score / 100.0)))

    return risk_score, confidence, detector_scores

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    }), 200

@app.route('/api/info', methods=['GET'])
def system_info():
    """System information"""
    return jsonify({
        'name': 'Automated Adversarial Monitoring and Self-Defense System',
        'version': '1.0.0',
        'description': '5-method heuristic ensemble with defense sanitization pipeline',
        'checkpoint_loaded': hybrid_detector.has_checkpoint,
        'cnn_checkpoint_loaded': hybrid_detector.has_checkpoint,
        'active_pipeline': 'Heuristic Detection + Sanitization' if not hybrid_detector.has_checkpoint else 'Hybrid Heuristic + CNN',
        'detection_methods': [
            'Robustness (confidence + stability)',
            'Frequency (texture analysis)',
            'Statistical (distribution analysis)',
            'Fingerprinting (entropy-based)',
            'Metadata (saturation check)'
        ]
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect_endpoint():
    """
    Main detection endpoint
    POST /api/detect with multipart image
    Returns: risk_score, decision, detector_scores, buffer_status
    """
    try:
        # Validate input
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Load and process image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Support model_type parameter (baseline vs robust)
        model_type = request.form.get('model_type', 'robust').lower()
        use_cnn = (model_type != 'baseline')

        # Run detection
        risk_score, confidence, detector_scores = detect_image_fast(image_array)

        # Determine decision (adjusted thresholds for better detection)
        if risk_score < 50:
            decision = 'ALLOW'
        elif risk_score < 75:
            decision = 'REVIEW'
        else:
            decision = 'BLOCK'

        # Update statistics
        state.total_detections += 1
        if decision == 'ALLOW':
            state.allow_count += 1
        elif decision == 'REVIEW':
            state.review_count += 1
            state.buffer_size += 1
            # Add to background retraining buffer
            adv_buffer.add(image_array, risk_score)
            logger.info(f"[BUFFER] Added adversarial image. Buffer: {adv_buffer.size()}/5")
        else:  # BLOCK
            state.block_count += 1
            state.buffer_size += 1
            # Add to background retraining buffer
            adv_buffer.add(image_array, risk_score)
            logger.info(f"[BUFFER] Added adversarial image. Buffer: {adv_buffer.size()}/5")

        # Auto-retrain trigger
        if state.buffer_size >= state.buffer_max:
            state.retrain_count += 1
            state.buffer_size = 0
            logger.info(f"Retrain #{state.retrain_count} triggered on {state.buffer_max} buffered adversarial samples.")

        # Build response
        response = {
            'success': True,
            'risk_score': float(risk_score),
            'decision': decision,
            'confidence': float(confidence),
            'checkpoint_loaded': hybrid_detector.has_checkpoint,
            'active_pipeline': "Heuristic Detection + Sanitization" if not hybrid_detector.has_checkpoint else "Hybrid Heuristic + CNN",
            'model_used': model_type,
            'detector_scores': detector_scores,
            'buffer_status': {
                'current_size': state.buffer_size,
                'max_size': state.buffer_max,
                'retrain_threshold': state.buffer_max,
                'percentage_full': (state.buffer_size / state.buffer_max * 100) if state.buffer_max > 0 else 0,
                'retrains_completed': state.retrain_count
            },
            'accuracy': {
                'baseline': state.baseline_accuracy,
                'current': state.current_accuracy,
                'improvement': (state.current_accuracy - state.baseline_accuracy) if (state.current_accuracy is not None and state.baseline_accuracy is not None) else 0.0,
                'benchmark_loaded': False
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Detection result: {image_file.filename} - Risk: {risk_score:.1f}%, Decision: {decision}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect_endpoint():
    """Batch detection for multiple images"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'Empty file list'}), 400

        results = []
        total_risk = 0.0

        for file in files:
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image = image.resize((32, 32), Image.Resampling.LANCZOS)
                image_array = np.array(image, dtype=np.float32) / 255.0

                risk_score, confidence, _ = detect_image_fast(image_array)

                if risk_score < 50:
                    decision = 'ALLOW'
                elif risk_score < 75:
                    decision = 'REVIEW'
                else:
                    decision = 'BLOCK'

                state.total_detections += 1
                total_risk += risk_score

                results.append({
                    'filename': secure_filename(file.filename),
                    'risk_score': float(risk_score),
                    'decision': decision,
                    'confidence': float(confidence),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Batch error on {file.filename}: {str(e)}")
                results.append({
                    'filename': secure_filename(file.filename),
                    'error': str(e),
                    'status': 'error'
                })

        avg_risk = total_risk / len([r for r in results if r.get('status') == 'success']) if results else 0

        return jsonify({
            'results': results,
            'summary': {
                'total_processed': len(results),
                'successful': len([r for r in results if r.get('status') == 'success']),
                'failed': len([r for r in results if r.get('status') == 'error']),
                'average_risk': float(avg_risk)
            }
        }), 200

    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        return jsonify({'error': f'Batch detection failed: {str(e)}'}), 500

@app.route('/api/detect-with-recovery', methods=['POST'])
def detect_with_recovery_endpoint():
    """
    Detection endpoint with automatic sanitization/recovery attempt
    POST /api/detect-with-recovery with multipart image

    If image is risky, tries to recover it using defense techniques
    Returns: original_risk, final_risk, decision, recovery_info
    """
    try:
        # Validate input
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Load and process image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Run detection WITH recovery pipeline
        pipeline = HybridDetectionWithRecovery(hybrid_detector)
        result = pipeline.detect_with_recovery(image_array)

        # Build response - convert all to native Python types (including nested dicts)
        recovery_info = result.get('recovery_info', {})
        explanation = result.get('explanation', {})

        # Clean explanation dict - convert numpy types
        clean_explanation = {}
        for key, value in explanation.items():
            if isinstance(value, dict):
                clean_explanation[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in value.items()}
            elif isinstance(value, (np.bool_, bool)):
                clean_explanation[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                clean_explanation[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean_explanation[key] = float(value)
            else:
                clean_explanation[key] = value

        response = {
            'success': True,
            'original_risk': float(result['original_risk']),
            'final_risk': float(result['final_risk']),
            'final_decision': str(result['final_decision']),
            'recovery': {
                'attempted': bool(result.get('recovery_attempted', False)),
                'successful': bool(result.get('recovery_successful', False)),
                'technique': str(recovery_info.get('technique_used', 'N/A')),
                'improvement': float(recovery_info.get('improvement', 0)),
            },
            'explanation': clean_explanation,
            'buffer_status': {
                'current_size': int(state.buffer_size),
                'max_size': int(state.buffer_max),
                'percentage_full': float((state.buffer_size / state.buffer_max * 100) if state.buffer_max > 0 else 0),
                'retrains_completed': int(state.retrain_count)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Log result
        if result['recovery_attempted']:
            if result['recovery_successful']:
                logger.info(
                    f"Recovery Success: {image_file.filename} "
                    f"({result['original_risk']:.1f}% -> {result['final_risk']:.1f}%) "
                    f"using {result['recovery_info']['technique_used']}"
                )
            else:
                logger.info(
                    f"Recovery Partial: {image_file.filename} "
                    f"({result['original_risk']:.1f}% -> {result['final_risk']:.1f}%)"
                )
        else:
            logger.info(f"No recovery needed: {image_file.filename} ({result['final_risk']:.1f}%)")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Detection with recovery error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics_endpoint():
    """Get system metrics including background retraining status"""
    trainer_status = background_trainer.get_status()

    improvement = (state.current_accuracy - state.baseline_accuracy) if (state.current_accuracy is not None and state.baseline_accuracy is not None) else 0.0
    return jsonify({
        'accuracy': {
            'baseline': state.baseline_accuracy,
            'current': state.current_accuracy,
            'improvement': improvement,
            'benchmark_loaded': False
        },
        'checkpoint_status': {
            'loaded': hybrid_detector.has_checkpoint,
            'mode': 'Stage 1 Heuristic Ensemble' if not hybrid_detector.has_checkpoint else 'Stage 1 + Stage 2 Hybrid'
        },
        'detections': {
            'total': state.total_detections,
            'allow': state.allow_count,
            'review': state.review_count,
            'block': state.block_count
        },
        'retraining': {
            'count': state.retrain_count,
            'buffer_size': state.buffer_size,
            'buffer_max': state.buffer_max,
            'background_retrains': trainer_status['retrain_count'],
            'is_retraining': trainer_status['is_retraining'],
            'bg_accuracy': trainer_status['current_accuracy']
        }
    }), 200

@app.route('/api/buffer-status', methods=['GET'])
def buffer_status_endpoint():
    """Get buffer status including background retraining buffer"""
    trainer_status = background_trainer.get_status()
    percentage = (state.buffer_size / state.buffer_max * 100) if state.buffer_max > 0 else 0

    return jsonify({
        'buffer_size': state.buffer_size,
        'buffer_max': state.buffer_max,
        'percentage_full': percentage,
        'retrain_threshold': state.buffer_max,
        'retrains_completed': state.retrain_count,
        'auto_retrain_enabled': True,
        'background_buffer': {
            'size': adv_buffer.size(),
            'max_size': 100,
            'is_retraining': trainer_status['is_retraining'],
            'background_retrains': trainer_status['retrain_count']
        }
    }), 200

@app.route('/api/stats', methods=['GET'])
def stats_endpoint():
    """Get detailed statistics"""
    total = state.total_detections
    if total > 0:
        allow_pct = (state.allow_count / total) * 100
        review_pct = (state.review_count / total) * 100
        block_pct = (state.block_count / total) * 100
    else:
        allow_pct = review_pct = block_pct = 0

    return jsonify({
        'total_detections': total,
        'distribution': {
            'allow': {'count': state.allow_count, 'percentage': allow_pct},
            'review': {'count': state.review_count, 'percentage': review_pct},
            'block': {'count': state.block_count, 'percentage': block_pct}
        },
        'accuracy': {
            'baseline': state.baseline_accuracy,
            'current': state.current_accuracy,
            'improvement': (state.current_accuracy - state.baseline_accuracy) if (state.current_accuracy is not None and state.baseline_accuracy is not None) else 0.0,
            'benchmark_loaded': False
        },
        'auto_learning': {
            'retrains_triggered': state.retrain_count,
            'buffer_status': f"{state.buffer_size}/{state.buffer_max}"
        }
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting AI Detection Firewall Backend...")
    logger.info("Running on http://0.0.0.0:9000")
    logger.info("Available endpoints:")
    logger.info("  /api/detect              - Detection only")
    logger.info("  /api/detect-with-recovery - Detection with auto-recovery attempt")
    logger.info("  /api/batch_detect        - Batch processing")
    logger.info("  /api/metrics             - System metrics")
    logger.info("  /api/buffer-status       - Buffer status")
    logger.info("  /api/stats               - Detailed statistics")
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True, use_reloader=False)
