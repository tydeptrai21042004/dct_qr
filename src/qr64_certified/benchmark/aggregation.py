from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_trials(trial_dir: Path) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for path in sorted(trial_dir.glob("*.json")):
        try:
            trials.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            trials.append({
                "status": "invalid_trial_file",
                "trial_file": path.name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "rows": [],
            })
    return trials


def flatten_rows(trials: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        rows.extend(dict(row) for row in trial.get("rows", []))
    return rows


def _finite_values(rows: Iterable[dict[str, Any]], field: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = _finite_values(rows, field)
    if values.size == 0:
        return {f"{field}_{name}": None for name in ("mean", "std", "median", "q10", "q90", "min", "max")} | {f"{field}_count": 0}
    return {
        f"{field}_count": int(values.size),
        f"{field}_mean": float(values.mean()),
        f"{field}_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        f"{field}_median": float(np.median(values)),
        f"{field}_q10": float(np.quantile(values, 0.10)),
        f"{field}_q90": float(np.quantile(values, 0.90)),
        f"{field}_min": float(values.min()),
        f"{field}_max": float(values.max()),
    }


def _summary_row(group_rows: list[dict[str, Any]], labels: dict[str, Any]) -> dict[str, Any]:
    ok_rows = [row for row in group_rows if row.get("status") == "ok"]
    result = {
        **labels,
        "row_count": len(group_rows),
        "success_count": len(ok_rows),
        "error_count": len(group_rows) - len(ok_rows),
        "success_rate": float(len(ok_rows) / max(len(group_rows), 1)),
    }
    fields = (
        "embedding_psnr_db", "embedding_ssim", "embedding_uiqi",
        "attacked_nc", "attacked_ncc", "attacked_ber", "attacked_bit_accuracy",
        "attacked_f1", "attacked_balanced_accuracy",
        "attacked_vs_host_psnr_db", "attacked_vs_host_ssim",
        "attack_damage_psnr_db", "extract_seconds", "embed_seconds", "key_size_bytes",
    )
    for field in fields:
        result.update(_stats(ok_rows, field))
    nc_values = _finite_values(ok_rows, "attacked_nc")
    accuracy_values = _finite_values(ok_rows, "attacked_bit_accuracy")
    result["nc_ge_0p90_rate"] = float(np.mean(nc_values >= 0.90)) if nc_values.size else None
    result["nc_ge_0p95_rate"] = float(np.mean(nc_values >= 0.95)) if nc_values.size else None
    result["bit_accuracy_ge_0p95_rate"] = float(np.mean(accuracy_values >= 0.95)) if accuracy_values.size else None
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_method_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_method_severity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_id = str(row.get("method_id", "unknown"))
        by_method[method_id].append(row)
        by_method_category[(method_id, str(row.get("attack_category", "unknown")))].append(row)
        by_method_severity[(method_id, str(row.get("attack_severity", "unknown")))].append(row)

    method_summary = [
        _summary_row(group, {
            "method_id": method_id,
            "method_kind": group[0].get("method_kind"),
            "display_name": group[0].get("display_name"),
            "blindness_tier": group[0].get("blindness_tier"),
            "fidelity_tier": group[0].get("fidelity_tier"),
            "comparison_group": group[0].get("comparison_group"),
            "payload_bits": group[0].get("payload_bits"),
        })
        for method_id, group in sorted(by_method.items())
    ]
    category_summary = [
        _summary_row(group, {"method_id": method_id, "attack_category": category})
        for (method_id, category), group in sorted(by_method_category.items())
    ]
    severity_summary = [
        _summary_row(group, {"method_id": method_id, "attack_severity": severity})
        for (method_id, severity), group in sorted(by_method_severity.items())
    ]
    return {
        "method_summary": method_summary,
        "category_summary": category_summary,
        "severity_summary": severity_summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate(trial_dir: Path, output_dir: Path, *, fingerprint: str | None = None) -> dict[str, Any]:
    trials = load_trials(trial_dir)
    if fingerprint is not None:
        trials = [t for t in trials if t.get("experiment_fingerprint") == fingerprint]
    rows = flatten_rows(trials)
    summaries = aggregate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rows.csv", rows)
    write_csv(output_dir / "method_summary.csv", summaries["method_summary"])
    write_csv(output_dir / "category_summary.csv", summaries["category_summary"])
    write_csv(output_dir / "severity_summary.csv", summaries["severity_summary"])
    report = {
        "trial_count": len(trials),
        "row_count": len(rows),
        "trial_status_counts": {
            status: sum(trial.get("status") == status for trial in trials)
            for status in sorted({str(trial.get("status")) for trial in trials})
        },
        **summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


__all__ = [
    "load_trials", "flatten_rows", "aggregate_rows", "write_csv", "write_aggregate",
]
