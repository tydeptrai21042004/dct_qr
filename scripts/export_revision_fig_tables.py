#!/usr/bin/env python3
"""Generate Table 10, Table 6 and Fig. 5 from ONE validated run.

Never replaces missing/failed experiments by zeros. No clean controls are in
revision_20x3_* suites. The default is strict publication mode.
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
from qr64_certified.benchmark.aggregation import load_trials  # noqa: E402
from qr64_certified.attacks import get_attack_suite  # noqa: E402


def dump(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def family(attack_id: str):
    # revision_sigma__gaussian__moderate -> gaussian
    bits = attack_id.split('__')
    if len(bits) != 3:
        raise ValueError(f'Not a revision attack id: {attack_id}')
    return bits[1]


def prepare(base: Path):
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    suite_id = manifest['protocol']['attack_suite']
    if suite_id not in ('revision_20x3_sigma', 'revision_20x3_variance'):
        raise ValueError('This report is only for a revision_20x3_sigma/variance run')
    protocol_id = manifest['protocol']['protocol_id']
    fingerprint = manifest['experiment_fingerprint']
    attack_ids = {a.attack_id for a in get_attack_suite(suite_id)}
    methods = [m['method_id'] for m in manifest['methods']]
    hosts = [Path(h['path']).name for h in manifest['hosts']]
    if any(Path(h).stem.casefold() in {'lena', 'lenna'} for h in hosts):
        raise ValueError('This revision excludes Lena/Lenna. Re-run without that host.')
    wms = [Path(w['path']).name for w in manifest['watermark_files']]
    seeds = [int(s) for s in manifest['protocol']['seeds']]
    expected_trial_ids = {(m, h, w, seed) for m in methods for h in hosts for w in wms for seed in seeds}
    trials = load_trials(base / 'trials')
    valid = []
    found = set()
    for trial in trials:
        identity = (trial.get('method_id'), trial.get('host_id'), trial.get('watermark_id'), trial.get('seed'))
        if identity not in expected_trial_ids:
            raise ValueError(f'Unexpected trial in directory: {identity}')
        if identity in found:
            raise ValueError(f'Duplicate trial: {identity}')
        found.add(identity)
        if trial.get('protocol_id') != protocol_id or trial.get('experiment_fingerprint') != fingerprint:
            raise ValueError(f'Stale or mismatched protocol/fingerprint for trial {identity}')
        if trial.get('status') != 'ok':
            raise ValueError(f'Failed embed or clean extraction for {identity}: {trial.get("error_message")}')
        rows = trial.get('rows')
        if not isinstance(rows, list) or {row.get('attack_id') for row in rows} != attack_ids or len(rows) != len(attack_ids):
            raise ValueError(f'Missing/duplicate attack records for {identity}')
        for row in rows:
            if row.get('status') != 'ok':
                raise ValueError(f'Failed attack {row.get("attack_id")} for {identity}')
            for metric in ('attacked_nc', 'attacked_ber', 'embedding_psnr_db', 'embedding_ssim'):
                if not isinstance(row.get(metric), (float, int)) or not np.isfinite(float(row[metric])):
                    raise ValueError(f'Missing/nonfinite {metric} for {identity}')
        valid.append(trial)
    missing = expected_trial_ids - found
    if missing:
        example = sorted(missing)[0]
        raise ValueError(f'{len(missing)} missing trials; first missing: {example}. ' +
                         'Finish run_unified_benchmark.py before publication export.')
    return manifest, valid


def write_report(base: Path, output: Path):
    manifest, trials = prepare(base)
    by_method = defaultdict(list)
    by_method_attack = defaultdict(list)
    by_family_severity = defaultdict(list)
    embed_quality = defaultdict(list)
    for trial in trials:
        mid = trial['method_id']
        embed_quality[mid].append(trial['embedding_metrics'])
        for row in trial['rows']:
            by_method[mid].append(row)
            by_method_attack[(mid, row['attack_id'])].append(row)
            if mid == 'dct_qr_theory':
                by_family_severity[(family(row['attack_id']), row['attack_severity'])].append(row)

    method_names = {m['method_id']: m.get('display_name', m['method_id']) for m in manifest['methods']}
    # Match the method order declared in the exact protocol (the agreed Table 10 order).
    table_order = [m["method_id"] for m in manifest["methods"]]
    t5 = []
    for mid in table_order:
        rows = by_method[mid]
        ncs = np.asarray([float(r['attacked_nc']) for r in rows])
        bers = np.asarray([float(r['attacked_ber']) for r in rows])
        embeds = embed_quality[mid]
        t5.append({'method_id': mid, 'display_name': method_names[mid],
                   'detector': next(m['blindness_tier'] for m in manifest['methods'] if m['method_id'] == mid),
                   'psnr_db': float(np.mean([e['psnr_db'] for e in embeds])),
                   'ssim': float(np.mean([e['ssim'] for e in embeds])),
                   'mean_nc': float(ncs.mean()), 'p5_nc': float(np.percentile(ncs, 5)),
                   'min_nc': float(ncs.min()), 'mean_ber': float(bers.mean()),
                   'case_count': len(rows)})
    dump(output/'table5_comparison.csv', t5,
         ['method_id','display_name','detector','psnr_db','ssim','mean_nc','p5_nc','min_nc','mean_ber','case_count'])
    # Same validated trial rows, optional table-number alias for the updated
    # manuscript.  Never insert zero-filled placeholder baselines.
    dump(output/'table10_equal_payload_comparison.csv', t5,
         ['method_id','display_name','detector','psnr_db','ssim','mean_nc','p5_nc','min_nc','mean_ber','case_count'])
    clean_rows = []
    for mid in table_order:
        native = [x for x in trials if x['method_id'] == mid]
        clean_rows.append({
            'method_id':mid,
            'mean_clean_nc':float(np.mean([x['clean_watermark_metrics']['nc'] for x in native])),
            'mean_clean_ber':float(np.mean([x['clean_watermark_metrics']['ber'] for x in native])),
            'clean_trial_count':len(native),
        })
    dump(output/'table10_clean_control_diagnostic.csv', clean_rows,
         ['method_id','mean_clean_nc','mean_clean_ber','clean_trial_count'])


    if 'dct_qr_theory' not in by_method:
        raise ValueError('Table 6 requires dct_qr_theory in the benchmark methods')
    t6=[]
    for a in get_attack_suite(manifest['protocol']['attack_suite']):
        rows = by_method_attack[('dct_qr_theory',a.attack_id)]
        t6.append({'attack_id':a.attack_id, 'family':family(a.attack_id),
                   'severity':a.severity, 'parameters':json.dumps(a.params,sort_keys=True),
                   'mean_nc':float(np.mean([r['attacked_nc'] for r in rows])),
                   'mean_ber':float(np.mean([r['attacked_ber'] for r in rows])),
                   'case_count':len(rows)})
    dump(output/'table6_by_attack_severity.csv',t6,
         ['attack_id','family','severity','parameters','mean_nc','mean_ber','case_count'])

    # Fig. 5 uses precisely the same trial rows as both tables, at the MODERATE
    # level only. This is a numerical comparison plot, not extracted-image panels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    moderate = [a for a in get_attack_suite(manifest['protocol']['attack_suite']) if a.severity == 'moderate']
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    x = np.arange(len(moderate))
    markers = ('o','s','^','D','v','x','P','*')
    for i, mid in enumerate(table_order):
        means = [np.mean([r['attacked_nc'] for r in by_method_attack[(mid,a.attack_id)]]) for a in moderate]
        ax.plot(x,means,marker=markers[i%len(markers)],lw=1.25,ms=3.5,label=method_names[mid])
    ax.set_xticks(x, [family(a.attack_id).replace('_',' ') for a in moderate], rotation=65, ha='right')
    ax.set_ylim(0,1.02)
    ax.set_ylabel('Mean NC (all hosts and watermarks)')
    ax.set_xlabel('Moderate attack condition')
    ax.grid(axis='y', alpha=.25)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output/'fig5_moderate_comparison.pdf')
    fig.savefig(output/'fig5_moderate_comparison.png',dpi=170)
    plt.close(fig)
    (output/'provenance.json').write_text(json.dumps({
         'protocol_id':manifest['protocol']['protocol_id'],
         'fingerprint':manifest['experiment_fingerprint'],
         'attack_suite':manifest['protocol']['attack_suite'],
         'methods':table_order,
         'host_count':len(manifest['hosts']),
         'watermark_count':len(manifest['watermark_files']),
         'seeds':manifest['protocol']['seeds'],
         'validated_trial_count':len(trials),
         'fig5_source':'same complete validated trial records, moderate severity only',
         'dct2019_disclosure':'64x64 adapted DCT implementation, key-assisted selected-block coordinates; not an exact 2019 reproduction',
    },indent=2)+'\n')
    print(f'Exported to {output}; {len(trials)} validated trials')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True, help='Directory containing manifest.json and trials/')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    write_report(a.run_dir, a.output)

if __name__ == '__main__':
    main()
