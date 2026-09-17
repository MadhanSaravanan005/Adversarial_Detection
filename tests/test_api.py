"""
Automated Test Suite for Adversarial Detection System
Tests API endpoints, image processing, detection, recovery, and error handling.
"""

import io
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app
from production_system.SANITIZATION_APPROACH import AdversarialImageSanitizer
from backend.hybrid_detector import HeuristicDetector


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create_test_image_bytes(color=(128, 128, 128), size=(32, 32), format="PNG"):
    """Helper to generate in-memory encoded image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


# ============================================================================
# 1. HEALTH & INFO ENDPOINTS
# ============================================================================

def test_health_endpoint(client):
    """Test GET /api/health responds with 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "uptime" in data
    assert "timestamp" in data


def test_system_info_endpoint(client):
    """Test GET /api/info returns system metadata."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.get_json()
    assert "name" in data
    assert "detection_methods" in data
    assert len(data["detection_methods"]) == 5


# ============================================================================
# 2. DETECTION ENDPOINT (VALID INPUTS)
# ============================================================================

def test_detect_clean_image(client):
    """Test POST /api/detect with a standard clean image."""
    img_bytes = create_test_image_bytes(color=(100, 100, 100))
    response = client.post(
        "/api/detect",
        data={"image": (img_bytes, "test_clean.png"), "model_type": "robust"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "risk_score" in data
    assert "decision" in data
    assert data["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
    assert "detector_scores" in data
    assert "buffer_status" in data
    assert data["buffer_status"]["retrain_threshold"] == 5


def test_detect_synthetic_adversarial_image(client):
    """Test POST /api/detect with high chromatic imbalance (adversarial-like)."""
    # High red, low green/blue triggers heuristic detection
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 250
    arr[:, :, 1] = 10
    arr[:, :, 2] = 10
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/detect",
        data={"image": (buf, "test_adv.png"), "model_type": "baseline"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["risk_score"] > 30.0


# ============================================================================
# 3. INVALID INPUT & ERROR HANDLING
# ============================================================================

def test_detect_no_image(client):
    """Test POST /api/detect with missing image field returns 400."""
    response = client.post("/api/detect", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_detect_empty_filename(client):
    """Test POST /api/detect with empty filename returns 400."""
    response = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_detect_corrupted_file(client):
    """Test POST /api/detect with non-image bytes returns 500 error cleanly."""
    response = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(b"not an image file"), "corrupt.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code in [400, 500]
    data = response.get_json()
    assert "error" in data


# ============================================================================
# 4. RECOVERY / SANITIZATION ENDPOINT
# ============================================================================

def test_recovery_endpoint(client):
    """Test POST /api/detect-with-recovery executes defense pipeline."""
    img_bytes = create_test_image_bytes(color=(220, 20, 20))
    response = client.post(
        "/api/detect-with-recovery",
        data={"image": (img_bytes, "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "original_risk" in data
    assert "final_risk" in data
    assert "final_decision" in data
    assert "recovery" in data
    assert "attempted" in data["recovery"]


# ============================================================================
# 5. METRICS & BUFFER ENDPOINTS
# ============================================================================

def test_metrics_endpoint(client):
    """Test GET /api/metrics returns system metrics structure."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "accuracy" in data
    assert "detections" in data
    assert "retraining" in data


def test_buffer_status_endpoint(client):
    """Test GET /api/buffer-status returns accurate buffer status."""
    response = client.get("/api/buffer-status")
    assert response.status_code == 200
    data = response.get_json()
    assert "buffer_size" in data
    assert "buffer_max" in data
    assert "auto_retrain_enabled" in data


# ============================================================================
# 6. SANITIZER UNIT TESTS
# ============================================================================

def test_sanitizer_transformations():
    """Verify all individual defensive transformations return valid arrays."""
    img = np.random.rand(32, 32, 3).astype(np.float32)

    jpeg = AdversarialImageSanitizer.jpeg_compression(img, quality=80)
    blur = AdversarialImageSanitizer.gaussian_blur(img, sigma=1.0)
    median = AdversarialImageSanitizer.median_filter(img, kernel_size=3)
    resize = AdversarialImageSanitizer.resizing_defense(img, downscale=0.5)
    bit = AdversarialImageSanitizer.bit_depth_reduction(img, bits=4)
    pipe = AdversarialImageSanitizer.apply_pipeline(img)

    for trans in [jpeg, blur, median, resize, bit, pipe]:
        assert trans.shape == (32, 32, 3)
        assert trans.dtype == np.float32
        assert trans.min() >= 0.0
        assert trans.max() <= 1.0


def test_heuristic_detector():
    """Verify HeuristicDetector computes scores within [0, 100]."""
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    risk, scores, is_certain = HeuristicDetector.detect(img)

    assert 0.0 <= risk <= 100.0
    assert len(scores) == 5
    assert isinstance(is_certain, bool)


def test_checkpoint_absence_handling(client):
    """Verify system explicitly declares checkpoint absence and falls back cleanly."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("checkpoint_loaded") is False
    assert "Heuristic" in data.get("active_pipeline", "")

    # Perform detection without checkpoint
    img_bytes = create_test_image_bytes(color=(50, 150, 200))
    det_resp = client.post(
        "/api/detect",
        data={"image": (img_bytes, "test_absence.png")},
        content_type="multipart/form-data"
    )
    assert det_resp.status_code == 200
    det_data = det_resp.get_json()
    assert det_data["success"] is True
    assert det_data["checkpoint_loaded"] is False
    assert 0.0 <= det_data["risk_score"] <= 100.0
    assert det_data["decision"] in ["ALLOW", "REVIEW", "BLOCK"]


def test_risk_score_bounds_and_decision_validity():
    """Verify risk scores across varied inputs are strictly bounded within [0, 100]."""
    for seed in [11, 42, 108, 999]:
        rng = np.random.default_rng(seed)
        img = rng.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
        risk, scores, _ = HeuristicDetector.detect(img)
        assert 0.0 <= risk <= 100.0
        for val in scores.values():
            assert 0.0 <= val <= 1.0
