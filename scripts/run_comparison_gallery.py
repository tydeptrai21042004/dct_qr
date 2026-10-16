#!/usr/bin/env python3
"""Side-by-side decoded watermark gallery: unchanged proposal and existing baselines.

Rows are attacks, columns are selected existing methods; each panel reports
NC/BER and the native payload size so incompatible payloads are not obscured.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from qr64_certified.benchmark import BenchmarkAdapter
from qr64_certified.benchmark.registry import resolve_methods
from qr64_certified.benchmark.adapters import load_baseline_parameters
from qr64_certified.attacks import get_attack_suite,apply_attack
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.common.metrics import watermark_metrics
from run_reviewer_diagnostics import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',default='dct_qr_theory,dct_dm_qim_chen2001_blind')
    p.add_argument('--host',type=Path,default=ROOT/'data/host/lenna.bmp')
    p.add_argument('--watermark',type=Path,default=ROOT/'data/watermark/wm.png')
    p.add_argument('--suite',default='common20')
    p.add_argument('--attack-limit',type=int,default=6)
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewer_comparison_gallery')
    a=p.parse_args()
    specs=resolve_methods(a.methods)
    attacks=list(get_attack_suite(a.suite))[:a.attack_limit]
    host=load_host_rgb(a.host)
    params=load_baseline_parameters(ROOT/'configs/benchmark/baseline_parameters.json')
    columns=[];rows=[]
    for spec in specs:
        adapter=BenchmarkAdapter(spec,ROOT,params.get(spec.method_id))
        wm=load_watermark_binary(a.watermark,size=spec.payload_size)
        col=[]
        try:
            image,key,_=adapter.embed(host,wm,seed=2026)
            for attack in attacks:
                row={'method_id':spec.method_id,'blindness_tier':spec.blindness_tier,
                     'payload_bits':wm.size,'attack_id':attack.attack_id,'status':'ok'}
                try:
                    recovered,_=adapter.extract(apply_attack(image,attack),key,
                         original_host=host if spec.requires_original_host else None)
                    row.update(watermark_metrics(wm,recovered))
                    preview=Image.fromarray((np.asarray(recovered)>0).astype(np.uint8)*255).resize((160,160),Image.Resampling.NEAREST).convert('RGB')
                except Exception as exc:
                    row.update(status='extract_error',error=f'{type(exc).__name__}: {exc}')
                    preview=Image.new('RGB',(160,160),'#ddd')
                col.append((row,preview));rows.append(row)
        except Exception as exc:
            for attack in attacks:
                row={'method_id':spec.method_id,'attack_id':attack.attack_id,'status':'embed_error',
                     'payload_bits':wm.size,'error':f'{type(exc).__name__}: {exc}'}
                col.append((row,Image.new('RGB',(160,160),'#ddd')));rows.append(row)
        columns.append(col)
    pw,ph=190,225
    sheet=Image.new('RGB',(pw*len(specs),ph*len(attacks)),'white')
    draw=ImageDraw.Draw(sheet)
    for j,spec in enumerate(specs):
        for i,(row,im) in enumerate(columns[j]):
            x,y=j*pw,i*ph
            sheet.paste(im,(x+15,y+48))
            draw.text((x+4,y+4),spec.method_id[:25],fill='black')
            draw.text((x+4,y+18),f"{row['payload_bits']} bits",fill='black')
            draw.text((x+4,y+31),str(row['attack_id'])[:28],fill='black')
            draw.text((x+4,y+212),f"NC={row.get('nc','ERR'):.4f}" if 'nc' in row else 'ERROR',fill='black')
    a.output.mkdir(parents=True,exist_ok=True)
    sheet.save(a.output/'method_attack_watermark_gallery.png')
    dump(a.output,'method_attack_gallery_metrics',rows,{'methods':[s.method_id for s in specs],
        'native_payload_sizes':{s.method_id:s.payload_bits for s in specs},
        'warning':'Different payload sizes/blindness tiers are explicitly labeled and not equivalent comparisons'})
    print('Saved',a.output)
if __name__=='__main__':main()
