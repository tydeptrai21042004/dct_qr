#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import signal
import platform
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.attacks import get_attack_suite, list_attack_suites  # noqa: E402
from qr64_certified.benchmark import (  # noqa: E402
    BenchmarkAdapter,
    BenchmarkProtocol,
    evaluate_trial,
    load_baseline_parameters,
    resolve_methods,
    write_aggregate,
)
from qr64_certified.common.io import list_image_files, load_host_rgb, load_watermark_binary  # noqa: E402


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load_protocol(path: Path) -> BenchmarkProtocol:
    return BenchmarkProtocol.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return values


# A successful process exit or trial.status == "ok" is not evidence that
# its metrics are complete. These fields are required for a publishable trial.
METRICS = ("embedding_metrics", "clean_watermark_metrics")
ROW_METRICS = ("attacked_nc", "attacked_ber", "attacked_bit_accuracy",
               "embedding_ssim", "embedding_psnr_db")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _trial_issues(payload: Any, *, protocol_id: str,
                  attack_ids: tuple[str, ...], fingerprint: str | None = None,
                  identity: tuple[str, str, str, int] | None = None) -> list[str]:
    if not isinstance(payload, dict):
        return ["invalid_json_object"]
    issues = []
    if payload.get("protocol_id") != protocol_id:
        issues.append("wrong_protocol")
    if fingerprint is not None and payload.get("experiment_fingerprint") != fingerprint:
        issues.append("wrong_fingerprint")
    if identity is not None and (payload.get("method_id"), payload.get("host_id"),
                                 payload.get("watermark_id"), payload.get("seed")) != identity:
        issues.append("wrong_identity")
    if payload.get("status") != "ok":
        issues.append("failed_trial")
    for group, fields in (("embedding_metrics", ("psnr_db", "ssim")),
                          ("clean_watermark_metrics", ("nc", "ber"))):
        metrics = payload.get(group)
        if not isinstance(metrics, dict) or any(not _finite_number(metrics.get(f)) for f in fields):
            issues.append(f"nonfinite_or_missing_{group}")
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) != len(attack_ids):
        issues.append("wrong_attack_count")
    if not isinstance(rows, list):
        return issues
    ids = [r.get("attack_id") for r in rows if isinstance(r, dict)]
    if len(ids) != len(rows) or tuple(ids) != attack_ids:
        issues.append("missing_duplicate_or_misordered_attacks")
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "ok":
            issues.append("failed_attack")
            continue
        if any(not _finite_number(row.get(f)) for f in ROW_METRICS):
            issues.append("nonfinite_or_missing_attack_metrics")
        if row.get("method_id") != payload.get("method_id") or row.get("seed") != payload.get("seed"):
            issues.append("wrong_attack_identity")
    return sorted(set(issues))


def _valid_trial(path: Path, *, protocol_id: str, attack_ids: tuple[str, ...],
                 fingerprint: str | None = None,
                 identity: tuple[str, str, str, int] | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return not _trial_issues(payload, protocol_id=protocol_id,
                                 attack_ids=attack_ids, fingerprint=fingerprint,
                                 identity=identity)
    except (OSError, json.JSONDecodeError):
        return False


def _environment() -> dict[str, Any]:
    import numpy
    import PIL
    import scipy
    import skimage

    data: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "Pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "JILP_NUM_THREADS": os.environ.get("JILP_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    try:
        import cv2
        data["opencv"] = cv2.__version__
    except Exception:
        data["opencv"] = None
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proposals and baselines through one attack/metric benchmark without changing method mathematics."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/benchmark/quick.json",
        help="Benchmark protocol JSON.",
    )
    parser.add_argument("--methods", help="Selector or comma-separated IDs; overrides protocol.")
    parser.add_argument("--suite", choices=tuple(list_attack_suites()), help="Attack suite override.")
    parser.add_argument("--host-limit", type=int, help="Number of sorted hosts to evaluate.")
    parser.add_argument("--attack-limit", type=int, help="Number of sorted suite attacks to evaluate.")
    parser.add_argument("--seeds", type=_parse_seed_list, help="Comma-separated embedding seeds.")
    parser.add_argument("--output-dir", type=Path, help="Root output directory override.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strict-clean-proposals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Finish the active trial, then checkpoint and stop at this session budget.")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Limit new trial attempts in this invocation (for smoke/CI tests).")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reattempt previously saved failures (off by default to avoid endless retries).")
    parser.add_argument(
        "--baseline-parameters",
        type=Path,
        default=ROOT / "configs/benchmark/baseline_parameters.json",
    )
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.max_trials is not None and args.max_trials <= 0:
        parser.error("--max-trials must be positive")

    os.environ.setdefault("JILP_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    protocol = _load_protocol(args.protocol)
    if args.methods is not None:
        protocol = replace(protocol, methods=args.methods)
    if args.suite is not None:
        protocol = replace(protocol, attack_suite=args.suite)
    if args.host_limit is not None:
        if args.host_limit <= 0:
            parser.error("--host-limit must be positive")
        protocol = replace(protocol, host_limit=args.host_limit)
    if args.attack_limit is not None:
        if args.attack_limit <= 0:
            parser.error("--attack-limit must be positive")
        protocol = replace(protocol, attack_limit=args.attack_limit)
    if args.seeds is not None:
        protocol = replace(protocol, seeds=args.seeds)
    if args.output_dir is not None:
        protocol = replace(protocol, output_dir=str(args.output_dir))
    if args.resume is not None:
        protocol = replace(protocol, resume=bool(args.resume))
    if args.strict_clean_proposals is not None:
        protocol = replace(protocol, strict_clean_proposals=bool(args.strict_clean_proposals))
    protocol.validate()

    methods = resolve_methods(protocol.methods)
    attacks = list(get_attack_suite(protocol.attack_suite))
    if protocol.attack_limit is not None:
        attacks = attacks[: protocol.attack_limit]

    host_dir = protocol.resolve_path(ROOT, protocol.host_dir)
    hosts = list_image_files(host_dir)
    if protocol.host_limit is not None:
        hosts = hosts[: protocol.host_limit]
    if not hosts:
        parser.error(f"No host images found under {host_dir}")

    watermark_paths = [protocol.resolve_path(ROOT, value) for value in protocol.watermarks]
    missing = [str(path) for path in watermark_paths if not path.exists()]
    if missing:
        parser.error(f"Missing watermark files: {', '.join(missing)}")

    base_output = protocol.resolve_path(ROOT, protocol.output_dir) / protocol.protocol_id
    trial_dir = base_output / "trials"
    aggregate_dir = base_output / "aggregate"
    trial_dir.mkdir(parents=True, exist_ok=True)
    baseline_params = load_baseline_parameters(args.baseline_parameters)

    manifest = {
        "protocol": {
            **protocol.__dict__,
            "watermarks": list(protocol.watermarks),
            "seeds": list(protocol.seeds),
            "methods": protocol.methods if isinstance(protocol.methods, str) else list(protocol.methods),
        },
        "methods": [spec.__dict__ for spec in methods],
        "attacks": [attack.__dict__ for attack in attacks],
        "hosts": [{"path": str(path.relative_to(ROOT)), "sha256": _hash_file(path)} for path in hosts],
        "watermark_files": [{"path": str(path.relative_to(ROOT)), "sha256": _hash_file(path)} for path in watermark_paths],
        "environment": _environment(),
        "protocol_file": str(args.protocol),
        "protocol_sha256": _hash_file(args.protocol),
    }
    # Hash configuration, selected files, attack parameters and implementation
    # source. Resume is never allowed across incompatible experiment versions.
    source_paths = sorted((ROOT / 'src' / 'qr64_certified').rglob('*.py'))
    source_paths += sorted((ROOT / 'scripts').glob('*.py'))
    manifest['source_sha256'] = {str(path.relative_to(ROOT)): _hash_file(path) for path in source_paths}
    manifest['baseline_parameters_sha256'] = _hash_file(args.baseline_parameters)
    fingerprint_basis = {k:v for k,v in manifest.items() if k not in ('environment', 'protocol_file')}
    # Output location and retry policy do not affect scientific measurements.
    # Exclude them so an identical experiment can resume from a restored ZIP
    # even if the notebook's working directory changes.
    fingerprint_basis['protocol'] = {k:v for k,v in manifest['protocol'].items()
                                     if k not in ('output_dir','resume','continue_on_error')}
    experiment_fingerprint = hashlib.sha256(json.dumps(fingerprint_basis, sort_keys=True).encode()).hexdigest()
    manifest['experiment_fingerprint'] = experiment_fingerprint
    previous_manifest = base_output / "manifest.json"
    if previous_manifest.exists():
        try:
            prior = json.loads(previous_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Invalid previous manifest, use a new output directory: {exc}")
        old_fp = prior.get("experiment_fingerprint")
        if old_fp != experiment_fingerprint:
            parser.error("Existing output directory belongs to a DIFFERENT source/configuration fingerprint. "
                         "Use a new output directory; never combine different method versions.")
    _write_atomic(previous_manifest, manifest)

    watermark_cache: dict[tuple[Path, int], Any] = {}
    expected_attack_ids = tuple(attack.attack_id for attack in attacks)
    total = len(methods) * len(hosts) * len(watermark_paths) * len(protocol.seeds)
    visited = 0
    skipped_ok = 0
    skipped_failed = 0
    attempted = 0
    started_all = time.perf_counter()
    graceful_stop = [False]
    def _request_stop(signum, frame):
        graceful_stop[0] = True
        print(f"Received signal {signum}: finishing current trial and preserving checkpoint", flush=True)
    # TERM is best effort; Kaggle may hard-kill the kernel instead.
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _request_stop)
        except (ValueError, OSError):
            pass
    stop_reason = "finished_all_requested_trials"

    def _save_progress(phase: str, filename: str | None = None) -> None:
        _write_atomic(base_output / "progress.json", {
            "phase": phase,
            "experiment_fingerprint": experiment_fingerprint,
            "protocol_id": protocol.protocol_id,
            "trials_expected": total,
            "trials_visited_this_session": visited,
            "trials_skipped_successful_this_session": skipped_ok,
            "trials_skipped_failed_this_session": skipped_failed,
            "trials_attempted_this_session": attempted,
            "last_trial": filename,
            "elapsed_seconds": float(time.perf_counter() - started_all),
        })

    _save_progress("running")
    for spec in methods:
        adapter = BenchmarkAdapter(spec, ROOT, baseline_params.get(spec.method_id))
        print(f"\n[{spec.method_kind}] {spec.method_id} | {spec.display_name}", flush=True)
        for host_path in hosts:
            host = load_host_rgb(host_path)
            for watermark_path in watermark_paths:
                cache_key = (watermark_path, spec.payload_size)
                if cache_key not in watermark_cache:
                    watermark_cache[cache_key] = load_watermark_binary(watermark_path, size=spec.payload_size)
                watermark = watermark_cache[cache_key]
                for seed in protocol.seeds:
                    filename = "__".join([
                        _slug(spec.method_id), _slug(host_path.stem),
                        _slug(watermark_path.stem), f"seed{seed}",
                    ]) + ".json"
                    output_path = trial_dir / filename
                    identity = (spec.method_id, host_path.name, watermark_path.name, seed)
                    if protocol.resume and output_path.exists():
                        try:
                            old_record = json.loads(output_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            old_record = None
                        if isinstance(old_record, dict) and (
                            old_record.get("experiment_fingerprint") == experiment_fingerprint
                            and old_record.get("protocol_id") == protocol.protocol_id
                            and (old_record.get("method_id"),old_record.get("host_id"),
                                 old_record.get("watermark_id"),old_record.get("seed")) == identity
                        ):
                            if not _trial_issues(old_record, protocol_id=protocol.protocol_id,
                                                  attack_ids=expected_attack_ids,
                                                  fingerprint=experiment_fingerprint, identity=identity):
                                skipped_ok += 1
                                visited += 1
                                continue
                            if not args.retry_failed:
                                skipped_failed += 1
                                visited += 1
                                continue
                    started = time.perf_counter()
                    trial = evaluate_trial(
                        adapter, host, watermark, attacks,
                        protocol_id=protocol.protocol_id,
                        host_id=host_path.name, watermark_id=watermark_path.name, seed=seed,
                        continue_on_error=protocol.continue_on_error,
                        strict_clean_proposal=protocol.strict_clean_proposals,
                    )
                    trial["experiment_fingerprint"] = experiment_fingerprint
                    trial["trial_seconds"] = float(time.perf_counter() - started)
                    trial["trial_file"] = filename
                    _write_atomic(output_path, trial)
                    attempted += 1
                    visited += 1
                    problems = _trial_issues(trial, protocol_id=protocol.protocol_id,
                                              attack_ids=expected_attack_ids,
                                              fingerprint=experiment_fingerprint, identity=identity)
                    print(f"  RUN {visited:04d}/{total:04d} {filename} "
                          f"status={trial.get('status')} audit={'PASS' if not problems else ','.join(problems)}",
                          flush=True)
                    _save_progress("running", filename)
                    if graceful_stop[0]:
                        stop_reason = "signal_requested"
                    elif args.max_seconds is not None and time.perf_counter() - started_all >= args.max_seconds:
                        stop_reason = "session_time_budget_reached"
                    elif args.max_trials is not None and attempted >= args.max_trials:
                        stop_reason = "session_trial_limit_reached"
                    if stop_reason != "finished_all_requested_trials":
                        break
                if stop_reason != "finished_all_requested_trials": break
            if stop_reason != "finished_all_requested_trials": break
        if stop_reason != "finished_all_requested_trials": break

    report = write_aggregate(trial_dir, aggregate_dir, fingerprint=experiment_fingerprint)
    expected_keys = {(s.method_id, h.name, w.name, int(seed)) for s in methods for h in hosts
                     for w in watermark_paths for seed in protocol.seeds}
    recorded: dict[tuple[str, str, str, int], list[str]] = {}
    for path in trial_dir.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("experiment_fingerprint") != experiment_fingerprint:
            continue
        key = (item.get("method_id"), item.get("host_id"), item.get("watermark_id"), item.get("seed"))
        if key not in expected_keys:
            continue
        recorded[key] = _trial_issues(item, protocol_id=protocol.protocol_id,
                                      attack_ids=expected_attack_ids,
                                      fingerprint=experiment_fingerprint, identity=key)
    valid = sum(not issues for issues in recorded.values())
    failed = len(recorded) - valid
    missing = len(expected_keys) - len(recorded)
    run_summary = {
        "trials_successful_with_complete_attacks": valid,
        "trials_failed_or_invalid": failed,
        "trials_missing": missing,
        "all_requested_trials_successful": valid == total and failed == 0 and missing == 0,
        "stop_reason": stop_reason,
        "experiment_fingerprint": experiment_fingerprint,
        "protocol_id": protocol.protocol_id,
        "trials_expected": total,
        "trials_visited_this_session": visited,
        "trials_attempted_this_session": attempted,
        "trials_skipped_successful_this_session": skipped_ok,
        "trials_skipped_failed_this_session": skipped_failed,
        "attack_rows_saved_including_partial": report["row_count"],
        "seconds_this_session": float(time.perf_counter() - started_all),
        "output_directory": str(base_output),
    }
    _write_atomic(base_output / "run_summary.json", run_summary)
    _save_progress("finished_session")
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
