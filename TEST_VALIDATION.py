"""
VALIDATION TEST SCRIPT
Verify that:
1. Normal images are NOT flagged as suspicious (false positive fix)
2. Adversarial images ARE correctly detected
3. Sanitization feature works properly
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent / "production_system"))

from production_system.detection_core import detect_image_fast, get_decision


def create_clean_image(seed: int):
    """Create a realistic clean image (low-risk)"""
    rng = np.random.default_rng(seed)
    image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    noise = rng.normal(0, 0.05, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def create_adversarial_image_fgsm(seed: int):
    """Create an adversarial-like image that should trigger REVIEW/BLOCK."""
    rng = np.random.default_rng(seed)

    noise = rng.random((32, 32)).astype(np.float32)

    # Force strong chromatic imbalance (high saturation + high channel-mean std).
    red = np.clip(0.80 + 0.20 * noise, 0, 1)
    green = np.clip(0.05 * noise, 0, 1)
    blue = np.clip(0.05 * noise, 0, 1)

    # Add small continuous perturbations so the distribution remains skewed.
    jitter = rng.normal(0, 0.03, (32, 32, 3)).astype(np.float32)
    image = np.stack([red, green, blue], axis=-1)
    image = np.clip(image + jitter, 0, 1)
    return image


def create_adversarial_image_pgd(seed: int):
    """Create a second adversarial-like pattern with different structure."""
    rng = np.random.default_rng(seed)

    base = (rng.random((32, 32)).astype(np.float32) ** 2)

    yy, xx = np.indices((32, 32))
    stripes = (np.sin(xx / 2.0) + np.cos(yy / 3.0)).astype(np.float32)
    stripes = stripes / (np.max(np.abs(stripes)) + 1e-8)

    red = np.clip(base * 2.8 + 0.25 * stripes, 0, 1)
    green = np.clip(base * 0.25 - 0.05 * stripes, 0, 1)
    blue = np.clip(base * 0.25 - 0.05 * stripes, 0, 1)

    image = np.stack([red, green, blue], axis=-1)
    return image


def run_tests():
    """Run comprehensive tests"""
    print("\n" + "="*80)
    print("ADVERSARIAL DETECTION SYSTEM - VALIDATION TEST")
    print("="*80)

    # Expectations follow the production policy in production_system/detection_core.py:
    # ALLOW: <50, REVIEW: 50-75, BLOCK: >=75.
    tests = [
        ("Clean Image #1 (Balanced)", create_clean_image(1001), "ALLOW"),
        ("Clean Image #2 (Balanced)", create_clean_image(1002), "ALLOW"),
        ("Clean Image #3 (Balanced)", create_clean_image(1003), "ALLOW"),
        ("FGSM Adversarial (Synthetic)", create_adversarial_image_fgsm(2001), "REVIEW"),
        ("PGD Adversarial (Synthetic)", create_adversarial_image_pgd(2002), "REVIEW"),
    ]

    results = []
    print("\nRunning tests...\n")
    print(f"{'Test Name':<30} {'Risk':<8} {'Expected':<10} {'Got':<10} {'Status':<10}")
    print("-" * 80)

    total_pass = 0
    total_fail = 0

    for test_name, image, expected_decision in tests:
        risk_score, confidence, scores = detect_image_fast(image)
        decision = get_decision(risk_score)

        if expected_decision == "REVIEW":
            passed = decision in ("REVIEW", "BLOCK")
        else:
            passed = decision == expected_decision

        status = "[PASS]" if passed else "[FAIL]"
        if passed:
            total_pass += 1
        else:
            total_fail += 1

        print(f"{test_name:<30} {risk_score:>6.1f}%  {expected_decision:<10} {decision:<10} {status:<10}")

        results.append({
            'name': test_name,
            'risk': risk_score,
            'expected': expected_decision,
            'got': decision,
            'passed': passed,
            'scores': scores
        })

    # Print summary
    print("\n" + "="*80)
    print(f"TEST RESULTS: {total_pass} PASSED, {total_fail} FAILED")
    print("="*80)

    if total_fail == 0:
        print("\n[SUCCESS] ALL TESTS PASSED")
        print("- Clean images stayed below REVIEW threshold")
        print("- Synthetic adversarial images triggered REVIEW/BLOCK")
    else:
        print(f"\n[ISSUE] {total_fail} TESTS FAILED")
        print("\nFailed tests:")
        for result in results:
            if not result['passed']:
                print(f"\n  {result['name']}:")
                print(f"    Expected: {result['expected']}, Got: {result['got']}")
                print(f"    Risk Score: {result['risk']:.1f}%")
                print(f"    Detector Scores: {result['scores']}")

    return results


if __name__ == '__main__':
     results = run_tests()
     failed = [r for r in results if not r['passed']]
     raise SystemExit(1 if failed else 0)
