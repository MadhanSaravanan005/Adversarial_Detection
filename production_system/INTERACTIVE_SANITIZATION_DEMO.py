"""
Interactive Adversarial Image Sanitization Demo
Demonstrates how defensive image transformations mitigate adversarial perturbations.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from production_system.SANITIZATION_APPROACH import AdversarialImageSanitizer
from production_system.detection_core import detect_image_fast, get_decision


def load_or_create_sample():
    """Load an existing test image or create a synthetic sample."""
    sample_files = [
        SCRIPT_DIR / "test_fgsm.jpg",
        SCRIPT_DIR / "test_pgd.jpg",
        SCRIPT_DIR / "test_clean.jpg",
        PROJECT_ROOT / "test_adversarial.jpg",
        PROJECT_ROOT / "test_clean.jpg",
    ]

    for p in sample_files:
        if p.exists():
            img = Image.open(p).convert("RGB").resize((32, 32))
            return np.array(img, dtype=np.float32) / 255.0, p.name

    # Fallback synthetic perturbed image
    rng = np.random.default_rng(42)
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    img[:, :, 0] += 0.35 * rng.random((32, 32))
    return np.clip(img, 0.0, 1.0), "synthetic_sample"


def run_demo():
    print("=" * 70)
    print("      ADVERSARIAL IMAGE SANITIZATION & DEFENSE DEMO")
    print("=" * 70)

    image_array, source_name = load_or_create_sample()
    init_risk, init_conf, _ = detect_image_fast(image_array)
    init_dec = get_decision(init_risk)

    print(f"\n[INPUT IMAGE] Source: {source_name}")
    print(f"  Initial Risk Score: {init_risk:.1f}%")
    print(f"  Initial Decision:   {init_dec}")
    print("\nApplying individual defensive transformations...\n")
    print(f"{'Defense Technique':<25} {'Post-Risk':<12} {'Delta':<12} {'Post-Decision':<15}")
    print("-" * 70)

    defenses = [
        ("JPEG (Quality=85)", lambda x: AdversarialImageSanitizer.jpeg_compression(x, quality=85)),
        ("JPEG (Quality=75)", lambda x: AdversarialImageSanitizer.jpeg_compression(x, quality=75)),
        ("Gaussian Blur (s=1.0)", lambda x: AdversarialImageSanitizer.gaussian_blur(x, sigma=1.0)),
        ("Median Filter (k=3)", lambda x: AdversarialImageSanitizer.median_filter(x, kernel_size=3)),
        ("Spatial Resizing (0.5x)", lambda x: AdversarialImageSanitizer.resizing_defense(x, downscale=0.5)),
        ("Bit-Depth (4-bit)", lambda x: AdversarialImageSanitizer.bit_depth_reduction(x, bits=4)),
        ("Full Pipeline (3-layer)", lambda x: AdversarialImageSanitizer.apply_pipeline(x)),
    ]

    out_dir = SCRIPT_DIR / "recovered_images"
    out_dir.mkdir(exist_ok=True)

    for name, func in defenses:
        recovered = func(image_array)
        risk, conf, _ = detect_image_fast(recovered)
        dec = get_decision(risk)
        delta = risk - init_risk
        delta_str = f"{delta:+.1f}%"

        print(f"{name:<25} {risk:>6.1f}%     {delta_str:>8}     {dec:<15}")

        # Save sample output
        safe_name = name.split()[0].lower() + "_recovered.jpg"
        out_path = out_dir / safe_name
        pil_out = Image.fromarray((recovered * 255).astype(np.uint8))
        pil_out.save(out_path)

    print("-" * 70)
    print(f"\nSanitized preview images saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
