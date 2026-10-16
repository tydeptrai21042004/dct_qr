#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def _valid_trial(path: Path, *, protocol_id: str, attack_ids: tuple[str, ...], fingerprint: str | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("protocol_id") != protocol_id:
            return False
        if "status" not in payload or "rows" not in payload:
            return False
        if payload.get("status") != "ok":
            return False
        if fingerprint is not None and payload.get("experiment_fingerprint") != fingerprint:
            return False
        row_ids = tuple(str(row.get("attack_id")) for row in payload.get("rows", []))
        return row_ids == attack_ids and all(row.get("status") == "ok" for row in payload.get("rows", []))
    except Exception:
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
    parser.add_argument(
        "--baseline-parameters",
        type=Path,
        default=ROOT / "configs/benchmark/baseline_parameters.json",
    )
    args = parser.parse_args()

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
    experiment_fingerprint = hashlib.sha256(json.dumps(fingerprint_basis, sort_keys=True).encode()).hexdigest()
    manifest['experiment_fingerprint'] = experiment_fingerprint
    _write_atomic(base_output / "manifest.json", manifest)

    watermark_cache: dict[tuple[Path, int], Any] = {}
    expected_attack_ids = tuple(attack.attack_id for attack in attacks)
    total = len(methods) * len(hosts) * len(watermark_paths) * len(protocol.seeds)
    completed = 0
    skipped = 0
    started_all = time.perf_counter()

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
                        _slug(spec.method_id),
                        _slug(host_path.stem),
                        _slug(watermark_path.stem),
                        f"seed{seed}",
                    ]) + ".json"
                    output_path = trial_dir / filename
                    if protocol.resume and output_path.exists() and _valid_trial(
                        output_path, protocol_id=protocol.protocol_id, attack_ids=expected_attack_ids,
                        fingerprint=experiment_fingerprint
                    ):
                        skipped += 1
                        completed += 1
                        print(f"  SKIP {completed:04d}/{total:04d} {filename}", flush=True)
                        continue

                    started = time.perf_counter()
                    trial = evaluate_trial(
                        adapter,
                        host,
                        watermark,
                        attacks,
                        protocol_id=protocol.protocol_id,
                        host_id=host_path.name,
                        watermark_id=watermark_path.name,
                        seed=seed,
                        continue_on_error=protocol.continue_on_error,
                        strict_clean_proposal=protocol.strict_clean_proposals,
                    )
                    trial["experiment_fingerprint"] = experiment_fingerprint
                    trial["trial_seconds"] = float(time.perf_counter() - started)
                    trial["trial_file"] = filename
                    _write_atomic(output_path, trial)
                    completed += 1
                    clean = trial.get("clean_watermark_metrics", {}).get("nc")
                    psnr_value = trial.get("embedding_metrics", {}).get("psnr_db")
                    print(
                        f"  RUN  {completed:04d}/{total:04d} {host_path.name:<16} "
                        f"wm={watermark_path.name:<22} seed={seed} status={trial.get('status')} "
                        f"PSNR={psnr_value} cleanNC={clean}",
                        flush=True,
                    )

    report = write_aggregate(trial_dir, aggregate_dir, fingerprint=experiment_fingerprint)
    # A completed loop is not equivalent to a scientifically complete benchmark.
    saved = []
    for path in trial_dir.glob('*.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('experiment_fingerprint') == experiment_fingerprint:
                saved.append(record)
        except (OSError, ValueError):
            pass
    valid = sum(t.get('status') == 'ok' and len(t.get('rows', [])) == len(attacks)
                and all(r.get('status') == 'ok' for r in t.get('rows', [])) for t in saved)
    run_summary = {
        'trials_successful_with_complete_attacks': valid,
        'all_requested_trials_successful': valid == total,
        'experiment_fingerprint': experiment_fingerprint,
        "protocol_id": protocol.protocol_id,
        "trials_expected": total,
        "trials_completed_or_skipped": completed,
        "trials_skipped": skipped,
        "attack_rows": report["row_count"],
        "seconds": float(time.perf_counter() - started_all),
        "output_directory": str(base_output),
    }
    _write_atomic(base_output / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
