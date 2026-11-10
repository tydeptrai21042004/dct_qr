#!/usr/bin/env python3
"""Additional descriptive and paired analyses for a COMPLETE revision 20x3 run.

Uses the exact validated trials underlying Table 10 / Table 6 / Fig. 5.
Does not retrain, retune, alter the proposed algorithm, or fabricate trials.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from export_revision_fig_tables import prepare, family  # noqa: E402
from qr64_certified.attacks import get_attack_suite  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Empty experiment output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(rows: list[dict]) -> dict:
    nc = np.asarray([float(r['attacked_nc']) for r in rows], dtype=float)
    ber = np.asarray([float(r['attacked_ber']) for r in rows], dtype=float)
    if len(nc) == 0 or not np.all(np.isfinite(nc)) or not np.all(np.isfinite(ber)):
        raise ValueError('Missing or nonfinite attack data')
    return {'n_cases': len(nc), 'mean_nc': float(nc.mean()),
            'median_nc': float(np.median(nc)), 'p5_nc': float(np.percentile(nc, 5)),
            'min_nc': float(nc.min()), 'mean_ber': float(ber.mean())}


def paired_bootstrap(deltas: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Bootstrap host-level means; keep watermark/attack observations in each host."""
    if deltas.ndim != 1 or deltas.size < 2 or not np.all(np.isfinite(deltas)):
        raise ValueError('Need >=2 complete finite independent host clusters')
    generator = np.random.default_rng(seed)
    sampled_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = generator.integers(0, len(deltas), size=len(deltas))
        sampled_means[i] = deltas[sample].mean()
    return (float(np.percentile(sampled_means, 2.5)),
            float(np.percentile(sampled_means, 97.5)))


def export(run_dir: Path, output_dir: Path, *, n_bootstrap: int = 2000, seed: int = 2026) -> dict:
    if n_bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    manifest, trials = prepare(run_dir)  # strict complete 20x3, excludes Lena/Lenna
    suite_name = manifest['protocol']['attack_suite']
    attacks = get_attack_suite(suite_name)
    attack_lookup = {a.attack_id: a for a in attacks}
    methods = {x['method_id']: x for x in manifest['methods']}
    units = defaultdict(dict)
    by_method_host = defaultdict(list)
    by_method_wm = defaultdict(list)
    by_method_family_severity = defaultdict(list)
    by_method_severity = defaultdict(list)
    by_method_gaussian = defaultdict(list)
    clean = []
    for t in trials:
        method = t['method_id']
        host, wm, trial_seed = t['host_id'], t['watermark_id'], int(t['seed'])
        key = (host, wm, trial_seed)
        if method in units[key]:
            raise ValueError('Duplicate trial identity')
        units[key][method] = t
        clean.append({'method_id': method, 'host': host, 'watermark': wm, 'seed': trial_seed,
                      'psnr_db': float(t['embedding_metrics']['psnr_db']),
                      'ssim': float(t['embedding_metrics']['ssim']),
                      'clean_nc': float(t['clean_watermark_metrics']['nc']),
                      'clean_ber': float(t['clean_watermark_metrics']['ber'])})
        for r in t['rows']:
            a = attack_lookup[r['attack_id']]
            by_method_host[method, host].append(r)
            by_method_wm[method, wm].append(r)
            by_method_family_severity[method, family(a.attack_id), a.severity].append(r)
            by_method_severity[method, a.severity].append(r)
            if family(a.attack_id) == 'gaussian':
                by_method_gaussian[method, a.severity].append(r)

    method_ids = sorted(methods)
    if any(set(x) != set(method_ids) for x in units.values()):
        raise ValueError('Not all methods have results for the same host-watermark-seed units')
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'clean_quality_per_pair.csv', clean)
    write_csv(output_dir / 'robustness_per_host.csv',
              [{'method_id': m, 'host': h, **describe(rs)} for (m, h), rs in sorted(by_method_host.items())])
    write_csv(output_dir / 'robustness_per_watermark.csv',
              [{'method_id': m, 'watermark': w, **describe(rs)} for (m, w), rs in sorted(by_method_wm.items())])
    by_fs = [{'method_id': m, 'family': f, 'severity': s, **describe(rs)}
             for (m, f, s), rs in sorted(by_method_family_severity.items())]
    write_csv(output_dir / 'robustness_by_family_severity.csv', by_fs)
    by_severity = [{'method_id': m, 'severity': s, **describe(rs)}
                   for (m, s), rs in sorted(by_method_severity.items())]
    write_csv(output_dir / 'robustness_by_severity.csv', by_severity)
    gauss = []
    for (m, s), rows in sorted(by_method_gaussian.items()):
        a = next(a for a in attacks if family(a.attack_id) == 'gaussian' and a.severity == s)
        var = (float(a.params['variance']) if a.group == 'gaussian_noise_var'
               else (float(a.params['sigma']) / 255.0) ** 2)
        gauss.append({'method_id': m, 'severity': s, 'gaussian_definition': a.group,
                      'variance_on_0_to_1_scale': var,
                      'nominal_sigma_on_8bit': 255.0 * np.sqrt(var), **describe(rows)})
    write_csv(output_dir / 'gaussian_sensitivity.csv', gauss)

    # Paired comparisons: aggregate 60 attacks per (host, watermark, seed).
    # Resample independent HOST clusters, keeping both watermarks together.
    paired = []
    if 'dct_qr_theory' in method_ids:
        for baseline in method_ids:
            if baseline == 'dct_qr_theory':
                continue
            by_host_nc, by_host_ber, by_host_psnr = defaultdict(list), defaultdict(list), defaultdict(list)
            for unit in sorted(units):
                trial_pro = units[unit]['dct_qr_theory']
                trial_baseline = units[unit][baseline]
                p = {r['attack_id']: r for r in trial_pro['rows']}
                b = {r['attack_id']: r for r in trial_baseline['rows']}
                if set(p) != set(b):
                    raise ValueError(f'Attack mismatch for {unit}')
                aids = sorted(p)
                host = unit[0]
                by_host_nc[host].append(np.mean([float(p[a]['attacked_nc']) - float(b[a]['attacked_nc']) for a in aids]))
                by_host_ber[host].append(np.mean([float(p[a]['attacked_ber']) - float(b[a]['attacked_ber']) for a in aids]))
                by_host_psnr[host].append(float(trial_pro['embedding_metrics']['psnr_db']) -
                                          float(trial_baseline['embedding_metrics']['psnr_db']))
            hosts = sorted(by_host_nc)
            nc_diffs = np.asarray([np.mean(by_host_nc[h]) for h in hosts])
            ber_diffs = np.asarray([np.mean(by_host_ber[h]) for h in hosts])
            psnr_diffs = np.asarray([np.mean(by_host_psnr[h]) for h in hosts])
            ncl, ncu = paired_bootstrap(nc_diffs, n_bootstrap=n_bootstrap, seed=seed)
            bel, beu = paired_bootstrap(ber_diffs, n_bootstrap=n_bootstrap, seed=seed + 1)
            paired.append({'baseline': baseline, 'baseline_detector': methods[baseline]['blindness_tier'],
                           'baseline_fidelity': methods[baseline]['fidelity_tier'],
                           'n_host_watermark_seed_units': len(units),
                           'n_independent_host_clusters':len(hosts),
                           'mean_nc_difference_proposed_minus_baseline': float(nc_diffs.mean()),
                           'nc_bootstrap95_low': ncl, 'nc_bootstrap95_high': ncu,
                           'mean_ber_difference_proposed_minus_baseline': float(ber_diffs.mean()),
                           'ber_bootstrap95_low': bel, 'ber_bootstrap95_high': beu,
                           'mean_psnr_difference_db': float(psnr_diffs.mean())})
        write_csv(output_dir / 'paired_comparisons_vs_proposed.csv', paired)

    # Fixed plots for the complete run, not cherry-picked 'best' attacks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    severity_order = ['mild', 'moderate', 'strong']
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method in method_ids:
        vals = [describe(by_method_severity[method, s])['mean_nc'] for s in severity_order]
        ax.plot(severity_order, vals, marker='o', label=methods[method].get('display_name', method))
    ax.set_ylabel('Mean NC across 20 attacks, 12 hosts, 2 watermarks')
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_robustness_vs_severity.pdf')
    fig.savefig(output_dir / 'fig_robustness_vs_severity.png', dpi=160)
    plt.close(fig)

    moderate_families = [family(a.attack_id) for a in attacks if a.severity == 'moderate']
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for method in method_ids:
        vals = [describe(by_method_family_severity[method, f, 'moderate'])['mean_nc'] for f in moderate_families]
        ax.plot(np.arange(len(moderate_families)), vals, marker='.', markersize=4,
                label=methods[method].get('display_name', method))
    ax.set_xticks(np.arange(len(moderate_families)), [f.replace('_', ' ') for f in moderate_families],
                  rotation=65, ha='right')
    ax.set_ylabel('Mean NC over 24 host-watermark pairs')
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_moderate_attack_breakdown.pdf')
    fig.savefig(output_dir / 'fig_moderate_attack_breakdown.png', dpi=160)
    plt.close(fig)

    meta = {'protocol_id': manifest['protocol']['protocol_id'],
            'source_fingerprint': manifest['experiment_fingerprint'],
            'attack_suite': suite_name, 'attack_count': len(attacks),
            'methods': method_ids, 'host_count': len(manifest['hosts']),
            'watermarks': [Path(w['path']).name for w in manifest['watermark_files']],
            'complete_trials': len(trials), 'attack_rows': sum(len(t['rows']) for t in trials),
            'bootstrap_replicates': n_bootstrap, 'bootstrap_seed': seed,
            'bootstrap_unit': 'host image; both watermarks, seeds and all attack conditions retained within host clusters',
            'interpretation': 'Paired confidence intervals describe host-watermark sampling uncertainty, not method fidelity or causal superiority.',
            'dct2019_note': 'Ariatmanto 2019 baseline is a key-assisted 4096-bit adaptation, not exact original implementation.'}
    (output_dir / 'experiment_provenance.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(trials)} complete trials, {len(attacks)} attacks, {len(method_ids)} methods -> {output_dir}')
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    export(args.run_dir, args.output, n_bootstrap=args.bootstrap, seed=args.seed)


if __name__ == '__main__':
    main()
