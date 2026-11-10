#!/usr/bin/env python3
"""Fig. 5: extracted watermark examples, verified against EXACT Table 5/6 trials.

The caption must identify the single host and watermark: these are examples,
not aggregate NC over 24 pairs. Refuses unvalidated or mismatched numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from export_revision_fig_tables import prepare, family  # noqa: E402
from qr64_certified.benchmark import BenchmarkAdapter  # noqa: E402
from qr64_certified.benchmark.registry import get_method_spec  # noqa: E402
from qr64_certified.benchmark.adapters import load_baseline_parameters  # noqa: E402
from qr64_certified.attacks import get_attack_suite, apply_attack  # noqa: E402
from qr64_certified.common.io import load_host_rgb, load_watermark_binary  # noqa: E402
from qr64_certified.common.metrics import watermark_metrics  # noqa: E402


def generate(run_dir, output, host_name, watermark_name, seed):
    manifest, trials = prepare(Path(run_dir))
    protocol = manifest['protocol']
    host_paths = {Path(h['path']).name: ROOT/h['path'] for h in manifest['hosts']}
    wm_paths = {Path(w['path']).name: ROOT/w['path'] for w in manifest['watermark_files']}
    if host_name not in host_paths or watermark_name not in wm_paths or seed not in protocol['seeds']:
        raise ValueError('Host/watermark/seed must belong to this exact validated protocol')
    method_specs = [get_method_spec(m['method_id']) for m in manifest['methods']]
    trial_lookup = {(t['method_id'],t['host_id'],t['watermark_id'],int(t['seed'])):t for t in trials}
    attacks = [a for a in get_attack_suite(protocol['attack_suite'])
               if a.severity == 'moderate' and family(a.attack_id) in ('gaussian', 'rotation', 'crop_zero_pad')]
    if len(attacks) != 3:
        raise ValueError(f'Expected exactly 3 moderate attacks, found {len(attacks)}')
    host = load_host_rgb(host_paths[host_name])
    params = load_baseline_parameters(ROOT/'configs/benchmark/baseline_parameters.json')
    cell_w,cell_h=200,194
    sheet = Image.new('RGB',(170+cell_w*len(method_specs),70+cell_h*len(attacks)),'white')
    pen=ImageDraw.Draw(sheet)
    pen.text((10,10),f'Host: {host_name} | Watermark: {watermark_name} | seed {seed}',fill='black')
    for j,spec in enumerate(method_specs):
        x=170+j*cell_w
        pen.text((x+8,40),spec.display_name[:28],fill='black')
        trial = trial_lookup[(spec.method_id,host_name,watermark_name,seed)]
        row_by_id = {r['attack_id']:r for r in trial['rows']}
        watermark = load_watermark_binary(wm_paths[watermark_name],size=spec.payload_size)
        adapter = BenchmarkAdapter(spec, ROOT, params.get(spec.method_id))
        marked,key,_ = adapter.embed(host, watermark, seed=seed)
        for i,a in enumerate(attacks):
            y=70+i*cell_h
            if j==0:
                pen.text((10,y+50),family(a.attack_id).replace('_',' '),fill='black')
                pen.text((10,y+68),str(a.params)[:26],fill='black')
            attacked = apply_attack(marked,a)
            recovered,_=adapter.extract(attacked,key,
                    original_host=host if spec.requires_original_host else None)
            measures = watermark_metrics(watermark,recovered)
            expected=row_by_id[a.attack_id]
            if abs(measures['nc']-expected['attacked_nc'])>1e-9 or abs(measures['ber']-expected['attacked_ber'])>1e-9:
                raise RuntimeError(f'Mismatch with saved benchmark: {spec.method_id} {a.attack_id}: '
                                   f'computed {measures["nc"]}/{measures["ber"]} vs '
                                   f'saved {expected["attacked_nc"]}/{expected["attacked_ber"]}')
            im = Image.fromarray((np.asarray(recovered)>0).astype(np.uint8)*255)
            im = im.resize((128,128),Image.Resampling.NEAREST).convert('RGB')
            sheet.paste(im,(x+35,y+5))
            pen.text((x+8,y+142),f'NC {measures["nc"]:.4f} | BER {measures["ber"]:.4f}',fill='black')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(output.with_suffix('.png'))
    sheet.save(output.with_suffix('.pdf'),'PDF',resolution=145)
    print(f'Saved {output.with_suffix(".pdf")} and PNG; every NC/BER checked against saved trial rows.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',default='pepper.bmp')
    p.add_argument('--watermark',default='Cc.logo.circle.png')
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    generate(a.run_dir,a.output,a.host,a.watermark,a.seed)

if __name__=='__main__':main()
