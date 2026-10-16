#!/usr/bin/env python3
"""Run full corrected DCT-QR theory and six one-factor ablations on one host."""
import argparse, json, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qr64_certified.proposals.dct_qr_theory import DCTQRTheoryConfig
from qr64_certified.proposals.dct_qr_theory import embed, extract
from qr64_certified.proposals.flags import ABLATION_TO_FIELD, apply_proposal_flags
from qr64_certified.attacks import get_attack_suite, apply_attack
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import psnr, ssim, nc, ber

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host', type=Path, default=ROOT/'data/host/lenna.bmp')
    parser.add_argument('--watermark', type=Path, default=ROOT/'data/watermark/wm.png')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dct_qr_theory_after_abc.json')
    parser.add_argument('--attack-suite',default='common20')
    parser.add_argument('--clean-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/theory_ablation')
    a=parser.parse_args()
    cfg=DCTQRTheoryConfig.from_mapping(json.loads(a.config.read_text()))
    host=load_host_rgb(a.host); watermark=load_watermark_binary(a.watermark,size=64)
    rows=[]
    for label in ('full',*ABLATION_TO_FIELD):
        c=cfg if label=='full' else apply_proposal_flags('dct_qr_theory',cfg,{'ablation':[label]})[0]
        im,key=embed(host,watermark,config=c)
        clean=extract(im,key)
        attacked=[]
        if not a.clean_only:
            for attack in get_attack_suite(a.attack_suite):
                try:
                    wm=extract(apply_attack(im,attack),key)
                    attacked.append({'attack':attack.attack_id,'nc':nc(watermark,wm),'ber':ber(watermark,wm)})
                except Exception as exc:
                    attacked.append({'attack':attack.attack_id,'error':str(exc)})
        valid=[r for r in attacked if 'nc' in r]
        row={'variant':label,'psnr':psnr(host,im),'ssim':ssim(host,im),
             'clean_nc':nc(watermark,clean),'clean_ber':ber(watermark,clean),
             'mean_attack_nc':float(sum(r['nc'] for r in valid)/len(valid)) if valid else None,
             'successful_attacks':len(valid),'total_attacks':len(attacked),'attacks':attacked}
        rows.append(row)
        print(label,'PSNR',round(row['psnr'],4),'cleanNC',round(row['clean_nc'],6),
              'meanNC',row['mean_attack_nc'],flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'ablation.json').write_text(json.dumps(rows,indent=2))
    with (a.output_dir/'ablation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='attacks'])
        writer.writeheader();writer.writerows([{k:v for k,v in row.items() if k!='attacks'} for row in rows])

if __name__=='__main__':main()
