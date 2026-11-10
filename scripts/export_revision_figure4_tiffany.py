#!/usr/bin/env python3
"""Figure 4: predeclared moderate attacks on Tiffany + C-logo, no Lena.

This output is an illustrative SINGLE host/watermark pair. It is not the mean of
Table 6. Every displayed NC/BER is asserted against the exact saved benchmark
row, and a complete strict revision benchmark is required before export.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from export_revision_fig_tables import prepare, family
from qr64_certified.attacks import get_attack_suite, apply_attack
from qr64_certified.benchmark import BenchmarkAdapter
from qr64_certified.benchmark.registry import get_method_spec
from qr64_certified.benchmark.adapters import load_baseline_parameters
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import watermark_metrics

# Set BEFORE inspecting NC to avoid selecting only impressive examples.
FAMILIES = (
    'jpeg', 'jpeg2000', 'bit_depth', 'color_quantization',
    'gaussian', 'salt_pepper', 'speckle', 'poisson',
    'median', 'gaussian_blur', 'rotation', 'crop_zero_pad',
)


def generate(run_dir: Path, output: Path, *, host_name: str = 'tiffany.bmp',
             watermark_name: str = 'Cc.logo.circle.png', seed: int = 2026):
    manifest, trials = prepare(run_dir)  # strict: no incomplete/mismatched data
    paths = {Path(h['path']).name: ROOT / h['path'] for h in manifest['hosts']}
    wm_paths = {Path(w['path']).name: ROOT / w['path'] for w in manifest['watermark_files']}
    if host_name.casefold() in {'lenna.bmp', 'lena.bmp'}:
        raise ValueError('Lena/Lenna hosts are excluded from this experiment')
    if host_name not in paths or watermark_name not in wm_paths or seed not in manifest['protocol']['seeds']:
        raise ValueError('Host/watermark/seed must appear in the exact validated run')
    lookup = {(t['method_id'],t['host_id'],t['watermark_id'],t['seed']):t for t in trials}
    key = ('dct_qr_theory', host_name, watermark_name, seed)
    if key not in lookup:
        raise ValueError('Proposed-method TIFFANY/C-logo trial not in completed run')
    trial = lookup[key]
    rows = {r['attack_id']: r for r in trial['rows']}
    suite = get_attack_suite(manifest['protocol']['attack_suite'])
    selected = [a for f in FAMILIES for a in suite if family(a.attack_id)==f and a.severity=='moderate']
    if len(selected) != len(FAMILIES):
        raise ValueError('The revision attack suite has changed; expected one moderate entry per chosen family')

    spec = get_method_spec('dct_qr_theory')
    adapter = BenchmarkAdapter(spec, ROOT,
        load_baseline_parameters(ROOT/'configs/benchmark/baseline_parameters.json').get(spec.method_id))
    host = load_host_rgb(paths[host_name])
    wm = load_watermark_binary(wm_paths[watermark_name], size=spec.payload_size)
    marked, key_obj, _ = adapter.embed(host, wm, seed=seed)

    # More readable than 12 compressed composite panels: show an attacked
    # Tiffany host thumbnail beside its recovered C-logo for every attack.
    cols, rows_count, panel_w, panel_h = 3, 4, 370, 225
    sheet = Image.new('RGB', (cols * panel_w, 65 + rows_count * panel_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((17, 16), 'Tiffany | C-logo | moderate attacks (selected in advance)', fill='black')
    draw.text((17, 36), 'Per-panel NC/BER: one host-watermark pair, not Table 6 averages', fill='black')
    metrics = []
    for i, a in enumerate(selected):
        attacked = apply_attack(marked, a)
        recovered, _ = adapter.extract(attacked, key_obj)
        values = watermark_metrics(wm, recovered)
        saved = rows[a.attack_id]
        for metric, col in (('nc', 'attacked_nc'), ('ber', 'attacked_ber')):
            if abs(float(values[metric]) - float(saved[col])) > 1e-9:
                raise RuntimeError(f'Cannot export inconsistent Fig. 4: {a.attack_id}, {metric}')
        family_name = family(a.attack_id)
        x = (i % cols) * panel_w
        y = 65 + (i // cols) * panel_h
        draw.text((x+11, y+7), f'{i+1:02d}  {family_name.replace("_", " ")}', fill='black')
        draw.text((x+11, y+27), str(dict(a.params)).replace("'seed': 123, ", '')[:50], fill='black')
        thumb = Image.fromarray(attacked).resize((152, 152), Image.Resampling.BICUBIC)
        logo = Image.fromarray((np.asarray(recovered)>0).astype(np.uint8)*255).resize(
            (152, 152), Image.Resampling.NEAREST).convert('RGB')
        sheet.paste(thumb, (x+17, y+48))
        sheet.paste(logo, (x+190, y+48))
        draw.text((x+19,y+204), f'NC={values["nc"]:.4f}  BER={values["ber"]:.4f}', fill='black')
        metrics.append({'host':host_name,'watermark':watermark_name,'seed':seed,
                        'method_id':'dct_qr_theory','attack_id':a.attack_id,
                        'parameters':json.dumps(a.params,sort_keys=True),
                        'severity':a.severity,'nc':values['nc'],'ber':values['ber']})
    output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(output.with_suffix('.png'))
    sheet.save(output.with_suffix('.pdf'), 'PDF', resolution=160)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    output.with_suffix('.json').write_text(json.dumps({
        'figure':'Figure 4', 'protocol_id':manifest['protocol']['protocol_id'],
        'experiment_fingerprint':manifest['experiment_fingerprint'],
        'host':host_name, 'watermark':watermark_name, 'seed':seed,
        'selection':'Predefined 12 attack families at moderate severity, not highest-NC filtering',
        'selection_families':FAMILIES,
        'statistics':'single host-watermark; not Table 6 means',
    },indent=2),encoding='utf-8')
    print(f'Saved {output.with_suffix(".pdf")}, PNG, CSV, and provenance JSON')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output',type=Path, required=True)
    p.add_argument('--host',default='tiffany.bmp')
    p.add_argument('--watermark',default='Cc.logo.circle.png')
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    generate(a.run_dir,a.output,host_name=a.host,watermark_name=a.watermark,seed=a.seed)


if __name__ == '__main__':
    main()
