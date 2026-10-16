#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import DCT_QR_THEORY, embed_proposal, extract_proposal
from qr64_certified.attacks import apply_attack
from qr64_certified.attacks.presets import COMMON_20, COMMON_20_GROUPS
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import ber, nc, psnr, ssim
from qr64_certified.proposals.dct_qr_theory import (
    _unlifted_analysis_matrices,
    adaptive_steps_from_beta,
    compute_theory_certificate,
    derived_eta,
    reference_qim_step,
)
from qr64_certified.proposals.dct_qr_theory import DCTQRTheoryConfig

def read_config(config_path, method_id):
    if method_id != DCT_QR_THEORY:
        raise ValueError("Only corrected DCT-QR theory supported")
    return DCTQRTheoryConfig.from_mapping(json.loads(Path(config_path).read_text(encoding="utf-8")))


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical illustrations for the DCT-QR theory proposal."
    )
    parser.add_argument("--host", type=Path, default=ROOT / "data" / "host" / "lenna.bmp")
    parser.add_argument("--watermark", type=Path, default=ROOT / "data" / "watermark" / "wm.png")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dct_qr_theory_after_abc.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "dct_qr_theory_numerics")
    args = parser.parse_args()

    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)
    cfg = read_config(args.config, DCT_QR_THEORY)

    eta = derived_eta()
    base = _unlifted_analysis_matrices(host, eta)
    certificate = compute_theory_certificate(host, eta=eta)
    spectral = np.linalg.norm(base, ord=2, axis=(1, 2))
    frobenius = np.linalg.norm(base, axis=(1, 2))
    shifted = base + certificate.lift * np.eye(4)[None, :, :]
    sigma_min = np.linalg.svd(shifted, compute_uv=False)[:, -1]
    step0 = reference_qim_step()
    steps = adaptive_steps_from_beta(certificate.beta, step0)

    watermarked, key, embed_meta = embed_proposal(
        DCT_QR_THEORY, host, watermark, config=cfg, return_metadata=True
    )
    clean, clean_meta = extract_proposal(watermarked, key, return_metadata=True)

    p = key.base_key.params
    payload_indices = np.asarray(p["payload_indices"], dtype=int)
    ref = np.asarray(p["decomposition_gain_reference"], dtype=np.float64)

    attack_rows: list[dict[str, object]] = []
    for attack in COMMON_20:
        attacked = apply_attack(watermarked, attack)
        recovered, meta = extract_proposal(attacked, key, return_metadata=True)
        # Recover full ratio diagnostics using the metadata already exposed by extraction.
        attack_rows.append({
            "attack": attack.attack_id,
            "category": attack.category,
            "severity": attack.severity,
            "nc": float(nc(watermark, recovered)),
            "ber": float(ber(watermark, recovered)),
            "gain_ratio_mean": float(meta.get("gain_ratio_mean", np.nan)),
            "gain_ratio_std": float(meta.get("gain_ratio_std", np.nan)),
            "gain_identity_error": float(meta.get("gain_identity_error", np.nan)),
            "inference_path": str(meta.get("inference_path", "")),
        })

    by_group: dict[str, dict[str, float | int]] = {}
    for group_name, attacks in COMMON_20_GROUPS.items():
        ids = {a.attack_id for a in attacks}
        rows = [r for r in attack_rows if r["attack"] in ids]
        by_group[group_name] = {
            "attack_count": len(rows),
            "mean_nc": float(np.mean([float(r["nc"]) for r in rows])),
            "min_nc": float(np.min([float(r["nc"]) for r in rows])),
            "mean_ber": float(np.mean([float(r["ber"]) for r in rows])),
        }

    summary = {
        "host": args.host.name,
        "watermark": args.watermark.name,
        "method": DCT_QR_THEORY,
        "attack_suite": "common20",
        "attack_count": len(COMMON_20),
        "eta": {
            "derived": eta,
            "luminance_norm": float(np.linalg.norm(np.array([0.299, 0.587, 0.114]))),
            "augmented_norm": float(np.linalg.norm(np.array([0.299 + eta, 0.587 - 0.5 * eta, 0.114 - 0.5 * eta]))),
        },
        "neumann_lift": {
            "lambda": float(certificate.lift),
            "max_spectral_norm_unlifted": float(np.max(spectral)),
            "max_frobenius_norm_unlifted": float(np.max(frobenius)),
            "spectral_ratio_max": float(np.max(spectral / certificate.lift)),
        },
        "beta_certificate": {
            "beta_min": float(np.min(certificate.beta)),
            "beta_median": float(np.median(certificate.beta)),
            "beta_max": float(np.max(certificate.beta)),
            "sigma_min_min": float(np.min(sigma_min)),
            "max_beta_over_sigma_min": float(np.max(certificate.beta / sigma_min)),
            "all_beta_le_sigma_min": bool(np.all(certificate.beta <= sigma_min + 1e-10)),
        },
        "adaptive_steps": {
            "reference_step": step0,
            "min": float(np.min(steps)),
            "q10": _pct(steps, 0.10),
            "median": float(np.median(steps)),
            "q90": _pct(steps, 0.90),
            "max": float(np.max(steps)),
        },
        "embedding": {
            "psnr": float(psnr(host, watermarked)),
            "ssim": float(ssim(host, watermarked)),
            "clean_nc": float(nc(watermark, clean)),
            "global_coset_flip": int(embed_meta["global_coset_flip"]),
            "projection_energy_ratio": float(embed_meta["coset"]["projection_energy_ratio"]),
            "integer_lattice_unit_updates": int(p["integer_lattice_unit_updates"]),
            "payload_blocks": int(payload_indices.size),
            "gain_reference_min": float(np.min(ref[payload_indices])),
            "gain_reference_median": float(np.median(ref[payload_indices])),
            "gain_reference_max": float(np.max(ref[payload_indices])),
            "clean_inference_path": clean_meta.get("inference_path", ""),
        },
        "common20": {
            "mean_nc": float(np.mean([float(r["nc"]) for r in attack_rows])),
            "q10_nc": float(np.quantile([float(r["nc"]) for r in attack_rows], 0.10)),
            "min_nc": float(np.min([float(r["nc"]) for r in attack_rows])),
            "mean_ber": float(np.mean([float(r["ber"]) for r in attack_rows])),
            "by_group": by_group,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "numerical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "common20_per_attack.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(attack_rows[0]))
        writer.writeheader()
        writer.writerows(attack_rows)

    lines = [
        "# DCT-QR theory numerical illustration",
        "",
        f"Host: `{args.host.name}`; watermark: `{args.watermark.name}`.",
        "",
        "## Theory diagnostics",
        "",
        f"- Derived opponent coefficient: `{eta:.9f}`.",
        f"- Norm preservation error: `{abs(summary['eta']['luminance_norm'] - summary['eta']['augmented_norm']):.3e}`.",
        f"- Neumann lift: `{certificate.lift:.6f}`; max spectral/lift ratio: `{summary['neumann_lift']['spectral_ratio_max']:.6f}` (< 1 is the sufficient condition).",
        f"- beta <= sigma_min for every block: `{summary['beta_certificate']['all_beta_le_sigma_min']}`; max beta/sigma_min: `{summary['beta_certificate']['max_beta_over_sigma_min']:.6f}`.",
        f"- Adaptive QIM step range: `{summary['adaptive_steps']['min']:.4f}` to `{summary['adaptive_steps']['max']:.4f}`; median `{summary['adaptive_steps']['median']:.4f}`.",
        "",
        "## Embedding and common-20 robustness",
        "",
        f"- PSNR: `{summary['embedding']['psnr']:.4f} dB`; SSIM: `{summary['embedding']['ssim']:.6f}`; clean NC: `{summary['embedding']['clean_nc']:.6f}`.",
        f"- Global-coset projection-energy ratio: `{summary['embedding']['projection_energy_ratio']:.6f}` (<= 1 by construction).",
        f"- Common-20 mean NC: `{summary['common20']['mean_nc']:.6f}`; q10 NC: `{summary['common20']['q10_nc']:.6f}`; minimum NC: `{summary['common20']['min_nc']:.6f}`.",
        "",
        "## Grouped robustness",
        "",
        "| Group | n | Mean NC | Min NC | Mean BER |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in by_group.items():
        lines.append(f"| {name} | {values['attack_count']} | {values['mean_nc']:.6f} | {values['min_nc']:.6f} | {values['mean_ber']:.6f} |")
    (args.output_dir / "NUMERICAL_ILLUSTRATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
