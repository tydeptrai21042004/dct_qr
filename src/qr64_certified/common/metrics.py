from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

_EPS = np.finfo(np.float64).eps


def _same_shape(a: Any, b: Any, name: str = "metric") -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        raise ValueError(f"{name} shape mismatch: {aa.shape} vs {bb.shape}")
    return aa, bb


def mse(a, b) -> float:
    a, b = _same_shape(a, b, "MSE")
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    return float(np.mean((af - bf) ** 2))


def rmse(a, b) -> float:
    return float(math.sqrt(mse(a, b)))


def mae(a, b) -> float:
    a, b = _same_shape(a, b, "MAE")
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def max_abs_error(a, b) -> float:
    a, b = _same_shape(a, b, "maximum absolute error")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a, b, data_range: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * math.log10((data_range**2) / m))


def snr(reference, test) -> float:
    reference, test = _same_shape(reference, test, "SNR")
    ref = reference.astype(np.float64)
    err = ref - test.astype(np.float64)
    signal_power = float(np.sum(ref**2))
    noise_power = float(np.sum(err**2))
    if noise_power <= _EPS:
        return float("inf")
    if signal_power <= _EPS:
        return float("-inf")
    return float(10.0 * math.log10(signal_power / noise_power))


def ssim(a, b, data_range: float = 255.0) -> float:
    a, b = _same_shape(a, b, "SSIM")
    if _ssim is not None:
        channel_axis = -1 if a.ndim == 3 else None
        return float(_ssim(a, b, data_range=data_range, channel_axis=channel_axis))
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = af.mean(), bf.mean()
    var_a, var_b = af.var(), bf.var()
    cov = ((af - mu_a) * (bf - mu_b)).mean()
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if abs(float(denominator)) <= _EPS:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denominator)


def uiqi(a, b) -> float:
    """Universal image quality index using the global Wang-Bovik form."""
    a, b = _same_shape(a, b, "UIQI")
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    ma, mb = float(af.mean()), float(bf.mean())
    va, vb = float(af.var()), float(bf.var())
    cov = float(np.mean((af - ma) * (bf - mb)))
    denominator = (va + vb) * (ma * ma + mb * mb)
    if abs(denominator) <= _EPS:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip((4.0 * cov * ma * mb) / denominator, -1.0, 1.0))


def correlation_coefficient(a, b) -> float:
    a, b = _same_shape(a, b, "correlation")
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denominator = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denominator <= _EPS:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(np.dot(af, bf) / denominator, -1.0, 1.0))


def normalized_absolute_error(reference, test) -> float:
    reference, test = _same_shape(reference, test, "NAE")
    ref = reference.astype(np.float64)
    numerator = float(np.sum(np.abs(ref - test.astype(np.float64))))
    denominator = float(np.sum(np.abs(ref)))
    if denominator <= _EPS:
        return 0.0 if numerator <= _EPS else float("inf")
    return float(numerator / denominator)


def structural_content(reference, test) -> float:
    reference, test = _same_shape(reference, test, "structural content")
    ref_energy = float(np.sum(reference.astype(np.float64) ** 2))
    test_energy = float(np.sum(test.astype(np.float64) ** 2))
    if test_energy <= _EPS:
        return 1.0 if ref_energy <= _EPS else float("inf")
    return float(ref_energy / test_energy)


def image_fidelity(reference, test) -> float:
    reference, test = _same_shape(reference, test, "image fidelity")
    ref = reference.astype(np.float64)
    error_energy = float(np.sum((ref - test.astype(np.float64)) ** 2))
    ref_energy = float(np.sum(ref**2))
    if ref_energy <= _EPS:
        return 1.0 if error_energy <= _EPS else float("-inf")
    return float(1.0 - error_energy / ref_energy)


def entropy(image) -> float:
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    probabilities = hist[hist > 0] / hist.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def histogram_intersection(a, b) -> float:
    a, b = _same_shape(a, b, "histogram intersection")
    au8 = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    bu8 = np.clip(np.rint(b), 0, 255).astype(np.uint8)
    scores: list[float] = []
    channels = au8.shape[-1] if au8.ndim == 3 else 1
    for channel in range(channels):
        xa = au8[..., channel] if channels > 1 else au8
        xb = bu8[..., channel] if channels > 1 else bu8
        ha = np.bincount(xa.ravel(), minlength=256).astype(np.float64)
        hb = np.bincount(xb.ravel(), minlength=256).astype(np.float64)
        ha /= max(float(ha.sum()), 1.0)
        hb /= max(float(hb.sum()), 1.0)
        scores.append(float(np.minimum(ha, hb).sum()))
    return float(np.mean(scores))


def edge_preservation_index(reference, test) -> float:
    """Correlation of Sobel edge magnitudes; one means identical edge structure."""
    reference, test = _same_shape(reference, test, "edge preservation")
    ref = reference.astype(np.float64)
    tst = test.astype(np.float64)
    if ref.ndim == 3:
        ref = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
        tst = 0.299 * tst[..., 0] + 0.587 * tst[..., 1] + 0.114 * tst[..., 2]
    if _ndi is not None:
        ref_mag = np.hypot(_ndi.sobel(ref, axis=0), _ndi.sobel(ref, axis=1))
        tst_mag = np.hypot(_ndi.sobel(tst, axis=0), _ndi.sobel(tst, axis=1))
    else:  # pragma: no cover
        ref_mag = np.hypot(*np.gradient(ref))
        tst_mag = np.hypot(*np.gradient(tst))
    return correlation_coefficient(ref_mag, tst_mag)


def to_bits(wm, threshold: int = 127) -> np.ndarray:
    arr = np.asarray(wm)
    if arr.size and float(np.max(arr)) <= 1.0:
        threshold = 0
    return (arr.astype(np.float64) > threshold).astype(np.uint8).ravel()


def nc(a, b) -> float:
    va = to_bits(a).astype(np.float64)
    vb = to_bits(b).astype(np.float64)
    if va.shape != vb.shape:
        raise ValueError(f"NC shape mismatch: {va.shape} vs {vb.shape}")
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    if den == 0:
        return 1.0 if np.array_equal(va, vb) else 0.0
    return float(np.clip(np.dot(va, vb) / den, -1.0, 1.0))


def ncc(a, b) -> float:
    va = to_bits(a).astype(np.float64)
    vb = to_bits(b).astype(np.float64)
    if va.shape != vb.shape:
        raise ValueError(f"NCC shape mismatch: {va.shape} vs {vb.shape}")
    va = va - va.mean()
    vb = vb - vb.mean()
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    if den == 0:
        return 1.0 if np.array_equal(va, vb) else 0.0
    return float(np.clip(np.dot(va, vb) / den, -1.0, 1.0))


def ber(a, b) -> float:
    va = to_bits(a)
    vb = to_bits(b)
    if va.shape != vb.shape:
        raise ValueError(f"BER shape mismatch: {va.shape} vs {vb.shape}")
    return float(np.mean(va != vb))


def bit_accuracy(a, b) -> float:
    return float(1.0 - ber(a, b))


def hamming_distance(a, b) -> int:
    va = to_bits(a)
    vb = to_bits(b)
    if va.shape != vb.shape:
        raise ValueError(f"Hamming shape mismatch: {va.shape} vs {vb.shape}")
    return int(np.count_nonzero(va != vb))


def confusion_counts(a, b) -> dict[str, int]:
    truth = to_bits(a)
    pred = to_bits(b)
    if truth.shape != pred.shape:
        raise ValueError(f"Confusion shape mismatch: {truth.shape} vs {pred.shape}")
    return {
        "tp": int(np.sum((truth == 1) & (pred == 1))),
        "tn": int(np.sum((truth == 0) & (pred == 0))),
        "fp": int(np.sum((truth == 0) & (pred == 1))),
        "fn": int(np.sum((truth == 1) & (pred == 0))),
    }


def watermark_metrics(reference, recovered) -> dict[str, float | int]:
    counts = confusion_counts(reference, recovered)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, _EPS)
    return {
        "nc": nc(reference, recovered),
        "ncc": ncc(reference, recovered),
        "ber": (fp + fn) / total,
        "bit_accuracy": (tp + tn) / total,
        "hamming_distance": fp + fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        **counts,
    }


def image_quality_metrics(reference, test, *, prefix: str = "") -> dict[str, float]:
    values = {
        "mse": mse(reference, test),
        "rmse": rmse(reference, test),
        "mae": mae(reference, test),
        "max_abs_error": max_abs_error(reference, test),
        "psnr_db": psnr(reference, test),
        "ssim": ssim(reference, test),
        "uiqi": uiqi(reference, test),
        "snr_db": snr(reference, test),
        "correlation": correlation_coefficient(reference, test),
        "nae": normalized_absolute_error(reference, test),
        "structural_content": structural_content(reference, test),
        "image_fidelity": image_fidelity(reference, test),
        "histogram_intersection": histogram_intersection(reference, test),
        "edge_preservation": edge_preservation_index(reference, test),
        "entropy_reference": entropy(reference),
        "entropy_test": entropy(test),
        "entropy_difference": abs(entropy(reference) - entropy(test)),
    }
    if not prefix:
        return values
    return {f"{prefix}{key}": value for key, value in values.items()}


def all_metrics(host, watermarked, watermark, extracted) -> dict[str, float | int]:
    """Backward-compatible combined metrics plus the expanded metric set."""
    result: dict[str, float | int] = {
        "psnr": psnr(host, watermarked),
        "ssim": ssim(host, watermarked),
        "nc": nc(watermark, extracted),
        "ncc": ncc(watermark, extracted),
        "ber": ber(watermark, extracted),
    }
    result.update(image_quality_metrics(host, watermarked, prefix="image_"))
    result.update({f"watermark_{k}": v for k, v in watermark_metrics(watermark, extracted).items()})
    return result


__all__ = [
    "mse", "rmse", "mae", "max_abs_error", "psnr", "snr", "ssim", "uiqi",
    "correlation_coefficient", "normalized_absolute_error", "structural_content",
    "image_fidelity", "entropy", "histogram_intersection", "edge_preservation_index",
    "to_bits", "nc", "ncc", "ber", "bit_accuracy", "hamming_distance",
    "confusion_counts", "watermark_metrics", "image_quality_metrics", "all_metrics",
]
