#!/usr/bin/env python3
"""Reviewer controls only; imports but never changes the DCT-QR proposal.

Modes: ablation, sensitivity, security, edge, timing, gallery.
Each trial is recorded even on failure; these tests are independent of the
publication benchmark and never substitute failures with successful values.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qr64_certified.attacks import apply_attack, get_attack_suite
from qr64_certified.benchmark import BenchmarkAdapter
from qr64_certified.benchmark.registry import proposal_spec, resolve_methods
from qr64_certified.common.io import list_image_files, load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import image_quality_metrics, watermark_metrics, to_bits
from qr64_certified.proposals.flags import apply_proposal_flags, ABLATION_TO_FIELD
from qr64_certified.proposals.dct_qr_theory import DCTQRTheoryConfig


def jsonable(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [jsonable(v) for v in x]
    return x


def dump(out: Path, name: str, rows: list[dict], info: dict):
    out.mkdir(parents=True,exist_ok=True)
    (out/f'{name}.json').write_text(json.dumps({'metadata':info,'rows':rows},indent=2,default=jsonable))
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with (out/f'{name}.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(jsonable(v)) if isinstance(v,(dict,list,tuple)) else jsonable(v) for k,v in r.items()})


def split_hosts(paths: list[Path], seed: int=2026, fraction: float=0.3):
    """Group at host-file level, so watermark, attack and seeds cannot leak."""
    ordered=sorted(paths,key=lambda p:hashlib.sha256(f'{seed}:{p.name}'.encode()).hexdigest())
    n_test=max(1,int(round(len(ordered)*fraction))) if len(ordered)>1 else 0
    return sorted(ordered[n_test:]),sorted(ordered[:n_test])


def load_inputs(args):
    hosts=list_image_files(ROOT/'data/host')
    train,test=split_hosts(hosts)
    (args.output/'host_split.json').parent.mkdir(parents=True,exist_ok=True)
    (args.output/'host_split.json').write_text(json.dumps({'split_seed':2026,'validation': [p.name for p in train], 'held_out':[p.name for p in test], 'unit':'unique host filename'},indent=2))
    selected={'validation':train,'test':test,'all':hosts}[args.partition]
    if args.host_limit: selected=selected[:args.host_limit]
    watermarks=sorted((ROOT/'data/watermark').glob('*.png'))
    if args.watermark_limit: watermarks=watermarks[:args.watermark_limit]
    return selected,watermarks


def adapter_with_config(config: DCTQRTheoryConfig) -> BenchmarkAdapter:
    adapter=BenchmarkAdapter(proposal_spec('dct_qr_theory'),ROOT)
    adapter._proposal_config=config
    return adapter


def summary_metadata(args):
    return {'mode':args.mode,'partition':args.partition,'suite':args.attack_suite,'seeds':args.seeds,
            'python':sys.version,'platform':platform.platform(), 'method_source_unchanged':True,
            'comparison_note':'Scientific configuration ablation/controls; this is not a new proposed method'}


class DiagnosticJournal:
    """Atomic per-combination cache so a killed Kaggle session loses at most
    the active image/method/parameter combination, never an entire grid.
    Changing experiment or code inputs rejects the prior journal.
    """
    def __init__(self, args, hosts, watermarks, attacks):
        source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'src'/'qr64_certified').rglob('*.py'))}
        source[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*hosts,*watermarks,ROOT/'configs/dct_qr_theory_after_abc.json']}
        params={k:v for k,v in vars(args).items() if k not in ('output','resume')}
        basis={'mode':args.mode,'params':params,
               'attack_specs':[a.__dict__ for a in attacks],
               'source_sha256':source,'input_sha256':inputs}
        self.signature=hashlib.sha256(json.dumps(basis,sort_keys=True,default=str).encode()).hexdigest()
        self.folder=args.output/(args.mode+'_chunks');self.folder.mkdir(parents=True,exist_ok=True)
        manifest=self.folder/'journal_manifest.json'
        if manifest.exists():
            old=json.loads(manifest.read_text())
            if old.get('signature')!=self.signature:
                raise ValueError('Cannot mix different experiment configurations or code versions in one diagnostic output. '
                                 'Use a new --output directory.')
        else:
            manifest.write_text(json.dumps({'signature':self.signature,'protocol':basis},indent=2,default=str))
        self.resume=args.resume

    def _file(self,key):
        digest=hashlib.sha256(json.dumps(key,sort_keys=True,default=str).encode()).hexdigest()
        return self.folder/(digest+'.json')

    def get(self,key):
        path=self._file(key)
        if not self.resume or not path.is_file():return None
        try:
            data=json.loads(path.read_text())
            if data.get('signature')==self.signature and data.get('key')==list(key) and data.get('complete') is True:
                return data['rows']
        except (OSError,ValueError,TypeError,KeyError):pass
        return None

    def save(self,key,rows):
        path=self._file(key)
        payload={'signature':self.signature,'key':list(key),'complete':True,'rows':jsonable(rows)}
        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        os.replace(tmp,path)


def run_ablation(args,hosts,watermarks,attacks):
    base=DCTQRTheoryConfig.from_mapping(json.loads((ROOT/'configs/dct_qr_theory_after_abc.json').read_text()))
    variants=[('full',())]+[(a,(a,)) for a in ABLATION_TO_FIELD]
    variants += [('uniform_step+no_gain_normalization',('uniform_step','no_gain_normalization')),
                 ('no_global_coset+no_spatial_icm',('no_global_coset','no_spatial_icm'))]
    rows=[]; journal=DiagnosticJournal(args,hosts,watermarks,attacks)
    for host_path, wm_path, seed, (name, switches) in itertools.product(hosts,watermarks,args.seeds,variants):
        job_key=(host_path.name,wm_path.name,seed,name)
        previous=journal.get(job_key)
        if previous is not None:
            rows.extend(previous);continue
        begin_rows=len(rows)
        row={'host':host_path.name,'watermark':wm_path.name,'seed':seed,'variant':name,'status':'ok'}
        try:
            c=apply_proposal_flags('dct_qr_theory',replace(base,seed=seed),{'ablation':switches})[0]
            adapter=adapter_with_config(c)
            host=load_host_rgb(host_path); wm=load_watermark_binary(wm_path,size=64)
            begin=time.perf_counter(); im,key,md=adapter.embed(host,wm,seed=seed)
            row['embed_seconds']=time.perf_counter()-begin
            row.update({f'embed_{k}':v for k,v in image_quality_metrics(host,im).items()})
            decoded,_=adapter.extract(im,key)
            row.update({f'clean_{k}':v for k,v in watermark_metrics(wm,decoded).items()})
            for attack in attacks:
                r=dict(row,attack_id=attack.attack_id,category=attack.category,severity=attack.severity)
                try:
                    begin=time.perf_counter()
                    decoded,_=adapter.extract(apply_attack(im,attack),key)
                    r['extract_seconds']=time.perf_counter()-begin
                    r.update({f'attacked_{k}':v for k,v in watermark_metrics(wm,decoded).items()})
                except Exception as exc: r.update(status='attack_error',error=str(exc))
                rows.append(r)
        except Exception as exc:
            rows.append(dict(row,status='embed_error',error=f'{type(exc).__name__}: {exc}'))
        journal.save(job_key,rows[begin_rows:])
        print('ablation',host_path.name,wm_path.name,seed,name,flush=True)
    dump(args.output,'ablation_by_attack',rows,summary_metadata(args))


def run_sensitivity(args,hosts,watermarks,attacks):
    base=DCTQRTheoryConfig.from_mapping(json.loads((ROOT/'configs/dct_qr_theory_after_abc.json').read_text()))
    # Explicitly pre-declared coupled grid; original default included.
    reference=[None,6.0,10.0]
    pilot_count=[64,128,256]
    pilot_step=[4.0,6.0,8.0]
    combos=list(itertools.product(reference,pilot_count,pilot_step))
    rows=[];journal=DiagnosticJournal(args,hosts,watermarks,attacks)
    for host_path,wm_path,seed,(step,n_pilots,pstep) in itertools.product(hosts,watermarks,args.seeds,combos):
        job_key=(host_path.name,wm_path.name,seed,step,n_pilots,pstep)
        previous=journal.get(job_key)
        if previous is not None:
            rows.extend(previous);continue
        begin_rows=len(rows)
        row={'host':host_path.name,'watermark':wm_path.name,'seed':seed,
             'reference_step':step,'pilot_count':n_pilots,'pilot_step':pstep,'status':'ok'}
        try:
            cfg=replace(base,seed=seed,reference_step=step,pilot_count=n_pilots,pilot_step=pstep).validated()
            adapter=adapter_with_config(cfg)
            host=load_host_rgb(host_path); wm=load_watermark_binary(wm_path,size=64)
            begin=time.perf_counter(); image,key,_=adapter.embed(host,wm,seed=seed)
            row['embed_seconds']=time.perf_counter()-begin
            row['psnr_db']=image_quality_metrics(host,image)['psnr_db']
            for attack in attacks:
                r=dict(row,attack_id=attack.attack_id,category=attack.category)
                try:
                    decoded,_=adapter.extract(apply_attack(image,attack),key)
                    r.update({f'attacked_{k}':v for k,v in watermark_metrics(wm,decoded).items()})
                except Exception as exc:r.update(status='attack_error',error=str(exc))
                rows.append(r)
        except Exception as exc:
            rows.append(dict(row,status='embed_error',error=f'{type(exc).__name__}: {exc}'))
        journal.save(job_key,rows[begin_rows:])
    dump(args.output,'coupled_parameter_sensitivity',rows,summary_metadata(args))


def run_security(args,hosts,watermarks,attacks):
    """Negative controls. 1-BER is a similarity score, NOT authentication.

    Same-host wrong-seed valid key, plain cover, and correct key positive.
    No forged cryptographic keyspace or encryption claims are made.
    """
    rows=[];journal=DiagnosticJournal(args,hosts,watermarks,attacks)
    for host_path,wm_path,seed in itertools.product(hosts,watermarks,args.seeds):
        job_key=(host_path.name,wm_path.name,seed,'security')
        previous=journal.get(job_key)
        if previous is not None:
            rows.extend(previous);continue
        begin_rows=len(rows)
        host=load_host_rgb(host_path); wm=load_watermark_binary(wm_path,size=64)
        a=BenchmarkAdapter(proposal_spec('dct_qr_theory'),ROOT)
        base={'host':host_path.name,'watermark':wm_path.name,'seed':seed}
        try:
            image,key,_=a.embed(host,wm,seed=seed)
            # Use separately generated valid key, not a malformed structure.
            _wrong_image,wrong_key,_=a.embed(host,wm,seed=seed+104729)
            tampered_indices=copy.deepcopy(key)
            if tampered_indices.base_key.params.get('payload_indices'):
                tampered_indices.base_key.params['payload_indices'] = list(
                    reversed(tampered_indices.base_key.params['payload_indices']))
            tampered_pilots=copy.deepcopy(key)
            if tampered_pilots.base_key.params.get('pilot_bits'):
                tampered_pilots.base_key.params['pilot_bits'] = [1-int(x) for x in
                    tampered_pilots.base_key.params['pilot_bits']]
            samples=[('positive_watermarked_correct_key',image,key,True),
                     ('negative_unwatermarked_correct_key',host,key,False),
                     ('negative_watermarked_wrong_seed_key',image,wrong_key,False),
                     ('negative_tampered_payload_schedule',image,tampered_indices,False),
                     ('negative_tampered_pilot_bits',image,tampered_pilots,False)]
            for case,observed,k,is_positive in samples:
                r=dict(base,case=case,positive=is_positive,status='ok')
                try:
                    decoded,_=a.extract(observed,k)
                    metrics=watermark_metrics(wm,decoded)
                    r.update(metrics)
                    r['similarity_score_1_minus_ber']=1.0-float(metrics['ber'])
                except Exception as exc:r.update(status='extract_error',error=f'{type(exc).__name__}: {exc}')
                rows.append(r)
        except Exception as exc:rows.append(dict(base,status='embed_error',error=f'{type(exc).__name__}: {exc}'))
        journal.save(job_key,rows[begin_rows:])
    dump(args.output,'negative_controls',rows,summary_metadata(args))
    # Only report empirical FPR/TPR at predeclared thresholds, no train/test leakage.
    good=[r for r in rows if r.get('status')=='ok']
    thresholds=(0.60,0.70,0.80,0.90,0.95,0.99)
    report=[]
    for t in thresholds:
        pos=[r for r in good if r['positive']]
        neg=[r for r in good if not r['positive']]
        report.append({'threshold_1_minus_ber':t,'positive_count':len(pos),'negative_count':len(neg),
            'TPR':sum(r['similarity_score_1_minus_ber']>=t for r in pos)/len(pos) if pos else None,
            'FPR':sum(r['similarity_score_1_minus_ber']>=t for r in neg)/len(neg) if neg else None})
    dump(args.output,'negative_control_roc_points',report,summary_metadata(args))


def run_edge(args):
    a=BenchmarkAdapter(proposal_spec('dct_qr_theory'),ROOT)
    wm=load_watermark_binary(ROOT/'data/watermark/wm.png',size=64)
    rng=np.random.default_rng(2026)
    hosts={
        'all_black':np.zeros((512,512,3),dtype=np.uint8),
        'all_gray':np.full((512,512,3),128,dtype=np.uint8),
        'all_white':np.full((512,512,3),255,dtype=np.uint8),
        'saturated_checkerboard':((np.indices((512,512)).sum(axis=0)%2)*255).astype(np.uint8)[:,:,None].repeat(3,axis=2),
        'near_dark':rng.integers(0,5,(512,512,3),dtype=np.uint8),
        'near_bright':rng.integers(250,256,(512,512,3),dtype=np.uint8),
    }
    rows=[]
    for name,host in hosts.items():
        r={'host_case':name,'status':'ok'}
        try:
            image,key,md=a.embed(host,wm,seed=args.seeds[0])
            decoded,_=a.extract(image,key)
            r.update({f'clean_{k}':v for k,v in watermark_metrics(wm,decoded).items()})
            r.update({f'embed_{k}':v for k,v in image_quality_metrics(host,image).items()})
            r['changed_channel_values']=int(np.count_nonzero(host!=image))
            r['pixels_at_low_endpoint']=int(np.count_nonzero(image==0))
            r['pixels_at_high_endpoint']=int(np.count_nonzero(image==255))
            r['embed_metadata']=md
        except Exception as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
        rows.append(r)
        print('edge',name,r['status'],flush=True)
    dump(args.output,'edge_case_audit',rows,summary_metadata(args))


def run_timing(args,hosts,watermarks,attacks):
    specs=resolve_methods(args.methods)
    rows=[];journal=DiagnosticJournal(args,hosts,watermarks,attacks)
    for spec in specs:
        a=BenchmarkAdapter(spec,ROOT)
        for host_path,wm_path in itertools.product(hosts,watermarks):
            host=load_host_rgb(host_path);wm=load_watermark_binary(wm_path,size=spec.payload_size)
            for repeat in range(args.timing_warmups+args.timing_repeats):
                job_key=(spec.method_id,host_path.name,wm_path.name,repeat,'timing')
                previous=journal.get(job_key)
                if previous is not None:
                    rows.extend(previous);continue
                row={'method_id':spec.method_id,'host':host_path.name,'watermark':wm_path.name,
                    'repeat':repeat,'phase':'warmup' if repeat<args.timing_warmups else 'timed',
                    'payload_bits':wm.size,'status':'ok'}
                try:
                    t=time.perf_counter();image,key,_=a.embed(host,wm,seed=args.seeds[0]);row['embed_seconds']=time.perf_counter()-t
                    t=time.perf_counter();decoded,_=a.extract(image,key,original_host=host if spec.requires_original_host else None)
                    row['extract_seconds']=time.perf_counter()-t
                    row['serial_embed_extract_fps']=1.0/(row['embed_seconds']+row['extract_seconds'])
                    row['quality_psnr_db']=image_quality_metrics(host,image)['psnr_db']
                    row['nc']=watermark_metrics(wm,decoded)['nc']
                except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                rows.append(row)
                journal.save(job_key,[row])
    dump(args.output,'runtime_wall_clock',rows,summary_metadata(args)|{
        'timing_includes':'embed and aligned clean extract, wall clock, excludes image loading',
        'fps_definition':'1/(embed_seconds+extract_seconds), serial still-image throughput; not video FPS',
        'jit_note':'Warmup trials excluded from timed summaries',})


def run_gallery(args,hosts,watermarks,attacks):
    spec=resolve_methods(args.methods)[0]
    a=BenchmarkAdapter(spec,ROOT)
    host=load_host_rgb(hosts[0]);wm=load_watermark_binary(watermarks[0],size=spec.payload_size)
    image,key,_=a.embed(host,wm,seed=args.seeds[0]); samples=[('clean',image)]
    for atk in attacks[:args.gallery_limit]:samples.append((atk.attack_id,apply_attack(image,atk)))
    tiles=[];rows=[]
    for attack_name,sample in samples:
        record={'attack':attack_name,'status':'ok'}
        try:
            decoded,_=a.extract(sample,key,original_host=host if spec.requires_original_host else None)
            record.update(watermark_metrics(wm,decoded))
            preview=Image.fromarray((np.asarray(decoded,dtype=np.uint8)>0).astype(np.uint8)*255).resize((160,160),Image.Resampling.NEAREST).convert('RGB')
        except Exception as exc:
            record.update(status='extract_error',error=str(exc))
            preview=Image.new('RGB',(160,160),'#ddd')
        tile=Image.new('RGB',(340,205),'white')
        thumb=Image.fromarray(sample).resize((160,160),Image.Resampling.BILINEAR)
        tile.paste(thumb,(5,25));tile.paste(preview,(175,25))
        draw=ImageDraw.Draw(tile);draw.text((5,5),attack_name[:47],fill='black');draw.text((5,187),f"NC={record.get('nc','ERR')} BER={record.get('ber','ERR')}",fill='black')
        tiles.append(tile);rows.append(record)
    ncols=3;nrows=(len(tiles)+ncols-1)//ncols
    sheet=Image.new('RGB',(ncols*340,nrows*205),'#eee')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%ncols)*340,(i//ncols)*205))
    args.output.mkdir(parents=True,exist_ok=True)
    sheet.save(args.output/'extraction_gallery.png')
    dump(args.output,'gallery_metrics',rows,summary_metadata(args))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('ablation','sensitivity','security','edge','timing','gallery'),required=True)
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True,
                   help='Resume atomic per-configuration chunks where supported.')
    p.add_argument('--partition',choices=('validation','test','all'),default='test')
    p.add_argument('--attack-suite',default='common20')
    p.add_argument('--host-limit',type=int)
    p.add_argument('--watermark-limit',type=int)
    p.add_argument('--attack-limit',type=int,default=None)
    p.add_argument('--methods',default='dct_qr_theory',help='For timing: existing methods; gallery: first selected')
    p.add_argument('--seeds',type=int,nargs='+',default=[2026])
    p.add_argument('--timing-warmups',type=int,default=1)
    p.add_argument('--timing-repeats',type=int,default=5)
    p.add_argument('--gallery-limit',type=int,default=8)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.timing_repeats<1 or args.timing_warmups<0:p.error('invalid timing repeats')
    args.output=args.output or ROOT/'results'/'reviewer_diagnostics'/args.mode
    hosts,watermarks=load_inputs(args)
    if not hosts and args.mode!='edge':p.error('No hosts for selected partition')
    attacks=list(get_attack_suite(args.attack_suite))
    if args.attack_limit is not None:attacks=attacks[:args.attack_limit]
    modes={'ablation':lambda:run_ablation(args,hosts,watermarks,attacks),
           'sensitivity':lambda:run_sensitivity(args,hosts,watermarks,attacks),
           'security':lambda:run_security(args,hosts,watermarks,attacks),
           'edge':lambda:run_edge(args),
           'timing':lambda:run_timing(args,hosts,watermarks,attacks),
           'gallery':lambda:run_gallery(args,hosts,watermarks,attacks)}
    modes[args.mode]()
    print('Saved diagnostic outputs to',args.output,flush=True)

if __name__=='__main__':main()
