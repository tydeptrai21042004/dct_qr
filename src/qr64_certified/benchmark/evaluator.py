from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qr64_certified.attacks import AttackConfig, apply_attack
from qr64_certified.common.metrics import image_quality_metrics, watermark_metrics

from .adapters import BenchmarkAdapter


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def key_size_bytes(key: Any) -> int | None:
    try:
        return int(len(pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception:
        return None


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def evaluate_trial(
    adapter: BenchmarkAdapter,
    host: np.ndarray,
    watermark: np.ndarray,
    attacks: Iterable[AttackConfig],
    *,
    protocol_id: str,
    host_id: str,
    watermark_id: str,
    seed: int,
    continue_on_error: bool = True,
    strict_clean_proposal: bool = False,
) -> dict[str, Any]:
    spec = adapter.spec
    host_u8 = np.asarray(host, dtype=np.uint8)
    watermark_u8 = np.asarray(watermark, dtype=np.uint8)
    trial: dict[str, Any] = {
        "protocol_id": protocol_id,
        "method_id": spec.method_id,
        "method_kind": spec.method_kind,
        "display_name": spec.display_name,
        "blindness_tier": spec.blindness_tier,
        "fidelity_tier": spec.fidelity_tier,
        "comparison_group": spec.comparison_group,
        "requires_original_host": spec.requires_original_host,
        "cover_dependent_key": spec.cover_dependent_key,
        "common_4096_payload": spec.common_4096_payload,
        "payload_size": [spec.payload_size, spec.payload_size],
        "payload_bits": int(watermark_u8.size),
        "payload_bpp": float(watermark_u8.size / (host_u8.shape[0] * host_u8.shape[1])),
        "host_id": host_id,
        "watermark_id": watermark_id,
        "seed": int(seed),
        "configuration": adapter.configuration_summary(seed),
        "status": "ok",
        "rows": [],
    }

    try:
        start = time.perf_counter()
        watermarked, key, embed_metadata = adapter.embed(host_u8, watermark_u8, seed=seed)
        embed_seconds = time.perf_counter() - start
        embedding_quality = image_quality_metrics(host_u8, watermarked)
        serialized_key_size = key_size_bytes(key)

        clean_start = time.perf_counter()
        recovered_clean, clean_metadata = adapter.extract(
            watermarked,
            key,
            original_host=host_u8 if spec.requires_original_host else None,
        )
        clean_extract_seconds = time.perf_counter() - clean_start
        clean_metrics = watermark_metrics(watermark_u8, recovered_clean)
        if strict_clean_proposal and spec.method_kind == "proposal" and float(clean_metrics["nc"]) < 1.0 - 1e-12:
            raise RuntimeError(f"Clean NC is {clean_metrics['nc']}, below the proposal requirement")

        trial.update({
            "embed_seconds": float(embed_seconds),
            "clean_extract_seconds": float(clean_extract_seconds),
            "key_size_bytes": serialized_key_size,
            "embedding_metrics": _json_safe(embedding_quality),
            "clean_watermark_metrics": _json_safe(clean_metrics),
            "embed_metadata": _json_safe(embed_metadata),
            "clean_extract_metadata": _json_safe(clean_metadata),
        })
    except Exception as exc:
        trial.update({
            "status": "embed_or_clean_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        if not continue_on_error:
            raise
        return _json_safe(trial)

    for attack in attacks:
        base_row: dict[str, Any] = {
            "protocol_id": protocol_id,
            "method_id": spec.method_id,
            "method_kind": spec.method_kind,
            "display_name": spec.display_name,
            "blindness_tier": spec.blindness_tier,
            "fidelity_tier": spec.fidelity_tier,
            "comparison_group": spec.comparison_group,
            "requires_original_host": spec.requires_original_host,
            "cover_dependent_key": spec.cover_dependent_key,
            "common_4096_payload": spec.common_4096_payload,
            "host_id": host_id,
            "watermark_id": watermark_id,
            "seed": int(seed),
            "payload_bits": int(watermark_u8.size),
            "payload_bpp": float(watermark_u8.size / (host_u8.shape[0] * host_u8.shape[1])),
            "attack_id": attack.attack_id,
            "attack_group": attack.group,
            "attack_category": attack.category,
            "attack_severity": attack.severity,
            "attack_params_json": json.dumps(attack.params, sort_keys=True),
            "status": "ok",
            "embed_seconds": trial.get("embed_seconds"),
            "clean_extract_seconds": trial.get("clean_extract_seconds"),
            "key_size_bytes": trial.get("key_size_bytes"),
            **_prefixed("embedding_", trial["embedding_metrics"]),
            **_prefixed("clean_", trial["clean_watermark_metrics"]),
        }
        try:
            attack_start = time.perf_counter()
            attacked = apply_attack(watermarked, attack)
            attack_seconds = time.perf_counter() - attack_start

            extract_start = time.perf_counter()
            recovered, metadata = adapter.extract(
                attacked,
                key,
                original_host=host_u8 if spec.requires_original_host else None,
            )
            extract_seconds = time.perf_counter() - extract_start

            base_row.update({
                "attack_seconds": float(attack_seconds),
                "extract_seconds": float(extract_seconds),
                **_prefixed("attacked_vs_host_", image_quality_metrics(host_u8, attacked)),
                **_prefixed("attack_damage_", image_quality_metrics(watermarked, attacked)),
                **_prefixed("attacked_", watermark_metrics(watermark_u8, recovered)),
                "extract_metadata_json": json.dumps(_json_safe(metadata), sort_keys=True),
            })
        except Exception as exc:
            base_row.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            if not continue_on_error:
                raise
        trial["rows"].append(_json_safe(base_row))

    trial["attack_count"] = len(trial["rows"])
    trial["successful_attacks"] = sum(row.get("status") == "ok" for row in trial["rows"])
    return _json_safe(trial)


__all__ = ["evaluate_trial", "key_size_bytes"]
