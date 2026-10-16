#!/usr/bin/env python3
"""Image-algorithm-through-a-frame-stream timing study. NO temporal voting/ECC.

Uses real input video when supplied, otherwise reproducible synthetic slideshow
from supplied BMPs. Every processed frame is independently embedded/extracted
using the unchanged 4096-bit DCT-QR Theory method.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from qr64_certified.benchmark import BenchmarkAdapter
from qr64_certified.benchmark.registry import proposal_spec
from qr64_certified.common.io import load_host_rgb,load_watermark_binary,list_image_files
from qr64_certified.common.metrics import watermark_metrics,image_quality_metrics
from qr64_certified.attacks import apply_attack
from qr64_certified.attacks.types import AttackConfig
from run_reviewer_diagnostics import dump


def to_frame_512(frame):
    """Center square crop followed by bicubic resize; log preprocessing."""
    h,w=frame.shape[:2]
    if min(h,w)<8:raise ValueError('Frame must be at least 8x8')
    s=min(h,w);y=(h-s)//2;x=(w-s)//2
    return cv2.resize(frame[y:y+s,x:x+s],(512,512),interpolation=cv2.INTER_CUBIC)


def frames(video:Path|None,max_frames:int):
    if video:
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise ValueError(f'Cannot open video {video}')
        nominal=float(cap.get(cv2.CAP_PROP_FPS))
        number=0
        try:
            while number<max_frames:
                ok,frame=cap.read()
                if not ok:break
                # OpenCV BGR -> RGB
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                yield number,to_frame_512(rgb),float(nominal) if nominal>0 else None
                number+=1
        finally:cap.release()
    else:
        hosts=list_image_files(ROOT/'data/host')
        for i in range(max_frames):
            rgb=load_host_rgb(hosts[i%len(hosts)])
            yield i,to_frame_512(rgb),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,help='Optional real video clip (not shipped)')
    p.add_argument('--max-frames',type=int,default=12)
    p.add_argument('--codec-quality',type=int,default=70)
    p.add_argument('--deadline-fps',type=float,default=30)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewer_frame_stream')
    args=p.parse_args()
    if args.max_frames<1 or args.deadline_fps<=0:p.error('Invalid frame count or deadline FPS')
    adapter=BenchmarkAdapter(proposal_spec('dct_qr_theory'),ROOT)
    watermark=load_watermark_binary(ROOT/'data/watermark/wm.png',size=64)
    attack=AttackConfig(f'jpeg_q{args.codec_quality}','jpeg',{'quality':args.codec_quality})
    rows=[]
    for index,frame,fps in frames(args.video,args.max_frames):
        row={'frame':index,'input_kind':'video' if args.video else 'synthetic_slideshow',
             'source_fps':fps,'frame_size_used':'512x512','status':'ok'}
        try:
            begin=time.perf_counter();im,key,_=adapter.embed(frame,watermark,seed=args.seed+index)
            row['embed_seconds']=time.perf_counter()-begin
            begin=time.perf_counter();attacked=apply_attack(im,attack)
            row['jpeg_seconds']=time.perf_counter()-begin
            begin=time.perf_counter();decoded,_=adapter.extract(attacked,key)
            row['extract_seconds']=time.perf_counter()-begin
            row['total_processing_seconds']=row['embed_seconds']+row['jpeg_seconds']+row['extract_seconds']
            row['on_time_30_fps']=row['total_processing_seconds']<=1.0/args.deadline_fps
            row['psnr_db']=image_quality_metrics(frame,im)['psnr_db']
            row.update({f'jpeg_{k}':v for k,v in watermark_metrics(watermark,decoded).items()})
        except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        print('frame',index,row['status'],flush=True)
    good=[r for r in rows if r['status']=='ok']
    latency=np.array([r['total_processing_seconds'] for r in good])
    summary={'source':'real_video' if args.video else 'synthetic_slideshow_NOT_video',
             'source_file':str(args.video) if args.video else None,
             'frames_observed':len(rows),'frames_successful':len(good),
             'jpeg_quality':args.codec_quality,'deadline_fps':args.deadline_fps,
             'preprocessing':'center crop square, resize to 512x512 via bicubic',
             'fidelity_warning':'Per-frame image embedding; no temporal voting, streaming queues or live camera capture',
             'mean_latency_ms':float(latency.mean()*1000) if len(latency) else None,
             'p95_latency_ms':float(np.percentile(latency,95)*1000) if len(latency) else None,
             'serial_effective_fps':float(len(good)/latency.sum()) if len(latency) else None,
             'on_time_ratio':float(np.mean([r['on_time_30_fps'] for r in good])) if good else None}
    dump(args.output,'frame_trials',rows,summary)
    (args.output/'frame_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
