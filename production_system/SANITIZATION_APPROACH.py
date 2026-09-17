"""
Adversarial Image Sanitization & Defense Module
Implements multi-layer defensive transformations to remove adversarial perturbations
from incoming images (JPEG compression, Gaussian smoothing, median filtering, resizing, bit-depth reduction).
"""

import io
import numpy as np
from PIL import Image
from scipy import ndimage


class AdversarialImageSanitizer:
    """
    Defensive transformations designed to disrupt adversarial perturbations
    while preserving underlying semantic image content.
    """

    @staticmethod
    def _to_uint8(image_array: np.ndarray) -> np.ndarray:
        """Helper to ensure image is in uint8 [0, 255] format."""
        arr = np.asarray(image_array, dtype=np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_float32(image_array: np.ndarray) -> np.ndarray:
        """Helper to ensure image is in float32 [0.0, 1.0] format."""
        arr = np.asarray(image_array, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def jpeg_compression(image_array: np.ndarray, quality: int = 75) -> np.ndarray:
        """
        JPEG Compression Defense:
        Compresses image using lossy DCT quantization to discard high-frequency adversarial noise.
        """
        uint8_img = AdversarialImageSanitizer._to_uint8(image_array)
        pil_img = Image.fromarray(uint8_img)

        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=int(quality))
        buffer.seek(0)

        compressed_img = Image.open(buffer).convert('RGB')
        return AdversarialImageSanitizer._to_float32(np.array(compressed_img))

    @staticmethod
    def gaussian_blur(image_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian Blur Defense:
        Applies continuous spatial smoothing across channels to dampen high-frequency perturbations.
        """
        float_img = AdversarialImageSanitizer._to_float32(image_array)
        smoothed = np.empty_like(float_img)
        for c in range(float_img.shape[2]):
            smoothed[:, :, c] = ndimage.gaussian_filter(float_img[:, :, c], sigma=float(sigma))

        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def median_filter(image_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Median Filter Defense:
        Non-linear edge-preserving filter that eliminates salt-and-pepper adversarial perturbations.
        """
        float_img = AdversarialImageSanitizer._to_float32(image_array)
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1

        filtered = np.empty_like(float_img)
        for c in range(float_img.shape[2]):
            filtered[:, :, c] = ndimage.median_filter(float_img[:, :, c], size=k)

        return np.clip(filtered, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def resizing_defense(image_array: np.ndarray, downscale: float = 0.5) -> np.ndarray:
        """
        Spatial Resizing Defense:
        Downsamples image to destroy pixel-aligned adversarial perturbations,
        then upsamples back to original dimensions using LANCZOS interpolation.
        """
        uint8_img = AdversarialImageSanitizer._to_uint8(image_array)
        h, w = uint8_img.shape[:2]

        new_w = max(4, int(w * float(downscale)))
        new_h = max(4, int(h * float(downscale)))

        pil_img = Image.fromarray(uint8_img)
        downsampled = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        upsampled = downsampled.resize((w, h), Image.Resampling.LANCZOS)

        return AdversarialImageSanitizer._to_float32(np.array(upsampled))

    @staticmethod
    def bit_depth_reduction(image_array: np.ndarray, bits: int = 4) -> np.ndarray:
        """
        Bit-Depth Reduction Defense:
        Quantizes pixel values to a reduced bit depth (e.g. 4 bits = 16 discrete levels),
        eliminating low-magnitude perturbations within quantization bins.
        """
        float_img = AdversarialImageSanitizer._to_float32(image_array)
        levels = float((1 << int(bits)) - 1)
        quantized = np.round(float_img * levels) / levels
        return np.clip(quantized, 0.0, 1.0).astype(np.float32)

    @classmethod
    def apply_pipeline(cls, image_array: np.ndarray, jpeg_quality: int = 85, blur_sigma: float = 1.0, median_kernel: int = 3) -> np.ndarray:
        """
        Sequential 3-layer defense pipeline matching the methodology:
        Raw -> JPEG Compression -> Gaussian Blur -> Median Filter
        """
        img = cls.jpeg_compression(image_array, quality=jpeg_quality)
        img = cls.gaussian_blur(img, sigma=blur_sigma)
        img = cls.median_filter(img, kernel_size=median_kernel)
        return img
