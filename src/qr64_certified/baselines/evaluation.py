from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from qr64_certified.attacks import AttackConfig, apply_attack
from qr64_certified.common.metrics import ber, image_quality_metrics, nc, ncc, psnr, ssim, watermark_metrics

from .registry import embed_baseline, extract_baseline, get_baseline_spec, normalize_baseline_id


def evaluate_baseline_attacks(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    attacks: Iterable[AttackConfig],
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Embed once, then evaluate one baseline under a deterministic attack suite.

    This utility intentionally performs no ranking across blindness tiers. It
    emits the canonical baseline ID and scientific metadata on every row so a
    caller cannot accidentally merge blind and non-blind methods silently.
    """
    canonical = normalize_baseline_id(method_id)
    spec = get_baseline_spec(canonical)
    host = np.asarray(host_rgb, dtype=np.uint8)
    watermark = np.asarray(watermark_binary, dtype=np.uint8)

    start = time.perf_counter()
    watermarked, key = embed_baseline(
        canonical,
        host,
        watermark,
        seed=seed,
        repeat=repeat,
        step=step,
        method_params=method_params,
    )
    embed_seconds = time.perf_counter() - start
    clean_psnr = psnr(host, watermarked)
    clean_quality = image_quality_metrics(host, watermarked)
    clean_start = time.perf_counter()
    clean_recovered = extract_baseline(
        watermarked, key, original_host=host if spec.requires_original_host else None
    )
    clean_extract_seconds = time.perf_counter() - clean_start
    clean_wm_metrics = watermark_metrics(watermark, clean_recovered)
    rows: list[dict[str, Any]] = []

    for attack in attacks:
        attack_start = time.perf_counter()
        try:
            attacked = apply_attack(watermarked, attack)
            attack_seconds = time.perf_counter() - attack_start
            extract_start = time.perf_counter()
            recovered = extract_baseline(
                attacked,
                key,
                original_host=host if spec.requires_original_host else None,
            )
            extract_seconds = time.perf_counter() - extract_start
            rows.append({
                "baseline_id": canonical,
                "display_name": spec.display_name,
                "blindness_tier": spec.blindness_tier,
                "fidelity_tier": spec.fidelity_tier,
                "attack_id": attack.attack_id,
                "attack_group": attack.group,
                "attack_category": attack.category,
                "attack_severity": attack.severity,
                "status": "ok",
                "clean_psnr_db": float(clean_psnr),
                "clean_ssim": float(clean_quality["ssim"]),
                "clean_uiqi": float(clean_quality["uiqi"]),
                "clean_nc": float(clean_wm_metrics["nc"]),
                "clean_ncc": float(clean_wm_metrics["ncc"]),
                "clean_ber": float(clean_wm_metrics["ber"]),
                "attacked_psnr_db": float(psnr(host, attacked)),
                "attacked_ssim": float(ssim(host, attacked)),
                "nc": float(nc(watermark, recovered)),
                "ncc": float(ncc(watermark, recovered)),
                "ber": float(ber(watermark, recovered)),
                **{f"watermark_{k}": v for k, v in watermark_metrics(watermark, recovered).items()},
                **{f"image_{k}": v for k, v in image_quality_metrics(host, attacked).items()},
                "embed_seconds": float(embed_seconds),
                "clean_extract_seconds": float(clean_extract_seconds),
                "attack_seconds": float(attack_seconds),
                "extract_seconds": float(extract_seconds),
            })
        except Exception as exc:
            row = {
                "baseline_id": canonical,
                "display_name": spec.display_name,
                "blindness_tier": spec.blindness_tier,
                "fidelity_tier": spec.fidelity_tier,
                "attack_id": attack.attack_id,
                "attack_group": attack.group,
                "attack_category": attack.category,
                "attack_severity": attack.severity,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "clean_psnr_db": float(clean_psnr),
                "embed_seconds": float(embed_seconds),
            }
            rows.append(row)
            if not continue_on_error:
                raise
    return rows


__all__ = ["evaluate_baseline_attacks"]
