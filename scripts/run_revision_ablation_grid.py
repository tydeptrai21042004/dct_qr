#!/usr/bin/env python3
"""Run all seven DCT--QR configurations on the same 12x2x60 revised test grid.

This invokes the existing, unchanged ablation code for each host/watermark.
All attack failures remain failures; no partial averages or missing-as-zero cells.
Result IDs include a protocol version. Abridged --limit-pairs runs are labeled
partial and cannot be presented as full-manuscript ablation evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qr64_certified.attacks import get_attack_suite
from qr64_certified.proposals.flags import ABLATION_TO_FIELD

VARIANTS = ('full', *ABLATION_TO_FIELD)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_result(source: Path, *, expected: set[str]) -> list[dict]:
    try:
        rows = json.loads(source.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return []
    if len(rows) != len(VARIANTS) or {row.get('variant') for row in rows} != set(VARIANTS):
        return []
    for item in rows:
        attacks = item.get('attacks', [])
        if (len(attacks) != len(expected)
                or {a.get('attack') for a in attacks} != expected
                or any('nc' not in a or 'ber' not in a for a in attacks)):
            return []
    return rows


def execute(protocol_path: Path, output: Path, limit_pairs: int | None = None) -> dict:
    proto = json.loads(protocol_path.read_text(encoding='utf-8'))
    if proto['attack_suite'] not in ('revision_20x3_variance', 'revision_20x3_sigma'):
        raise ValueError('Expected one of the exact revision 20x3 suites')
    if proto.get('seeds') != [2026]:
        raise ValueError('The ablation grid requires the fixed paper seed 2026')
    hosts = proto['host_files']
    watermarks = proto['watermarks']
    if (len(hosts) != 12 or len(watermarks) != 2
            or any('lena' in h.lower() or 'lenna' in h.lower() for h in hosts)):
        raise ValueError('Full paper ablation must use the 12 specified non-Lena hosts and two watermarks')
    pairs = [(h, w) for h in hosts for w in watermarks]
    if limit_pairs is not None:
        if limit_pairs < 1:
            raise ValueError('--limit-pairs must be positive')
        pairs = pairs[:limit_pairs]
    attacks = get_attack_suite(proto['attack_suite'])
    attack_ids = {a.attack_id for a in attacks}
    assert len(attacks) == 60 and len(attack_ids) == 60
    source_file = ROOT / 'src/qr64_certified/proposals/dct_qr_theory.py'
    source_hash = _hash(source_file)
    attack_hashes = {str(x.relative_to(ROOT)): _hash(x) for x in (
        ROOT/'src/qr64_certified/attacks/presets.py',
        ROOT/'src/qr64_certified/attacks/operations.py',
        ROOT/'src/qr64_certified/proposals/flags.py',
        ROOT/'scripts/run_theory_ablation.py',
        ROOT/'configs/dct_qr_theory_after_abc.json',
    )}
    result_root = output / proto['protocol_id']
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / 'manifest.json'
    manifest = {'protocol_id': proto['protocol_id'], 'source_sha256': source_hash,
                'supporting_source_sha256':attack_hashes,
                'protocol_sha256': _hash(protocol_path), 'attack_suite': proto['attack_suite'],
                'variants': VARIANTS, 'host_count': len(hosts),
                'watermark_count': len(watermarks), 'pair_count': len(pairs),
                'attack_count': 60}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old != manifest:
            raise ValueError('Existing ablation outputs have a different source/configuration; use a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')

    all_rows=[]
    for host, wm in pairs:
        name = f'{Path(host).stem}__{Path(wm).stem}'
        subdir = result_root / 'pairs' / name
        output_json = subdir / 'ablation.json'
        rows = _safe_result(output_json, expected=attack_ids) if output_json.is_file() else []
        if not rows:
            command = [sys.executable, str(ROOT/'scripts/run_theory_ablation.py'),
                       '--host', str(ROOT/'data/host'/host),
                       '--watermark', str(ROOT/wm),
                       '--attack-suite', proto['attack_suite'],
                       '--output-dir', str(subdir)]
            print('RUN:', host, Path(wm).name, flush=True)
            env=os.environ.copy()
            for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','JILP_NUM_THREADS'):
                env[key]='1'
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            rows = _safe_result(output_json, expected=attack_ids)
            if not rows:
                raise RuntimeError(f'Incomplete attacks in {output_json}; no summary will be produced')
        for record in rows:
            all_rows.append({'variant':record['variant'],'host':host,'watermark':Path(wm).name,
                             'psnr_db':float(record['psnr']),'ssim':float(record['ssim']),
                             'clean_nc':float(record['clean_nc']), 'clean_ber':float(record['clean_ber']),
                             'mean_nc':float(record['mean_attack_nc']),
                             'mean_ber':sum(float(a['ber']) for a in record['attacks'])/60.0})
    # Only publish a summary of completed full-grid runs. Partial runs are
    # clearly named and cannot be mistaken for paper's 24-pair results.
    complete=(len(pairs)==24 and len(all_rows)==len(VARIANTS)*24)
    summary=[]
    for variant in VARIANTS:
        records=[r for r in all_rows if r['variant']==variant]
        if len(records)!=len(pairs):
            raise RuntimeError(f'Incomplete variant: {variant}')
        summary.append({'variant':variant, 'n_pairs':len(records),
                        'mean_psnr_db':sum(x['psnr_db'] for x in records)/len(records),
                        'mean_ssim':sum(x['ssim'] for x in records)/len(records),
                        'mean_clean_nc':sum(x['clean_nc'] for x in records)/len(records),
                        'mean_attack_nc':sum(x['mean_nc'] for x in records)/len(records),
                        'mean_attack_ber':sum(x['mean_ber'] for x in records)/len(records),
                        'complete_24_pair_paper_grid':complete})
    with (result_root/'per_pair.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    with (result_root/'summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (result_root/'status.json').write_text(json.dumps({'complete':complete,
                'measured_pairs':len(pairs), 'required_pairs':24,
                'suite':proto['attack_suite'],'warning':'Partial runs are diagnostic only' if not complete else ''},indent=2)+'\n')
    print(f'Ablation: {len(pairs)}/24 pairs, complete={complete} -> {result_root}')
    return {'complete':complete,'rows':all_rows,'summary':summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/benchmark/revision_20x3_variance.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/revision_ablation')
    parser.add_argument('--limit-pairs',type=int,help='Optional smoke; partial results are labeled')
    opts=parser.parse_args()
    execute(opts.protocol.resolve(),opts.output.resolve(),opts.limit_pairs)


if __name__=='__main__':
    main()
