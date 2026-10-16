#!/usr/bin/env python3
"""Validation-only strength calibration, independent held-out matched-PSNR test.

Uses the SAME corrected proposal and SAME baseline implementations. A baseline
that ignores `step` is explicitly marked uncalibratable, never silently shown
as matched. Failed/infeasible runs are retained in output.
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from qr64_certified.benchmark import BenchmarkAdapter
from qr64_certified.benchmark.registry import resolve_methods
from qr64_certified.benchmark.types import BaselineRunParameters
from qr64_certified.benchmark.adapters import load_baseline_parameters
from qr64_certified.attacks import get_attack_suite,apply_attack
from qr64_certified.common.io import load_host_rgb,load_watermark_binary,list_image_files
from qr64_certified.common.metrics import image_quality_metrics,watermark_metrics
from run_reviewer_diagnostics import split_hosts,dump


def configured_adapter(spec,strength,default_params):
    params=default_params.get(spec.method_id,BaselineRunParameters())
    if spec.method_kind=='proposal':
        adapter=BenchmarkAdapter(spec,ROOT)
        adapter._proposal_config=replace(adapter._proposal_config, reference_step=strength).validated()
    else:
        adapter=BenchmarkAdapter(spec,ROOT,replace(params,step=strength))
    return adapter


def median_psnr(adapter,hosts,watermark,seed):
    measured=[]
    for path in hosts:
        try:
            host=load_host_rgb(path)
            wm=load_watermark_binary(watermark,size=adapter.spec.payload_size)
            image,key,_=adapter.embed(host,wm,seed=seed)
            v=image_quality_metrics(host,image)['psnr_db']
            if np.isfinite(v):measured.append(float(v))
        except Exception:
            return None
    return float(np.median(measured)) if len(measured)==len(hosts) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',default='common_4096')
    p.add_argument('--targets',nargs='+',type=float,default=[50.0,52.0,54.0])
    p.add_argument('--tolerance',type=float,default=0.75)
    p.add_argument('--suite',default='common20')
    p.add_argument('--attack-limit',type=int)
    p.add_argument('--validation-limit',type=int,default=None)
    p.add_argument('--test-limit',type=int,default=None)
    p.add_argument('--strengths',nargs='+',type=float,default=[2,4,6,8,10,12,16,20,26,34])
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewer_matched_psnr')
    args=p.parse_args()
    if args.tolerance<=0:p.error('tolerance must be positive')
    train,test=split_hosts(list_image_files(ROOT/'data/host'))
    if args.validation_limit:train=train[:args.validation_limit]
    if args.test_limit:test=test[:args.test_limit]
    if not train or not test:p.error('Need independent validation and test host groups')
    wm_paths=sorted((ROOT/'data/watermark').glob('*.png'))
    params=load_baseline_parameters(ROOT/'configs/benchmark/baseline_parameters.json')
    specs=resolve_methods(args.methods)
    attacks=get_attack_suite(args.suite)
    if args.attack_limit:attacks=attacks[:args.attack_limit]
    calibration=[];test_rows=[]
    for spec in specs:
        for target in args.targets:
            candidates=[]
            for strength in args.strengths:
                adapter=configured_adapter(spec,strength,params)
                value=median_psnr(adapter,train,wm_paths[0],2026)
                candidates.append({'strength':strength,'validation_psnr_db':value})
            valid=[c for c in candidates if c['validation_psnr_db'] is not None]
            chosen=min(valid,key=lambda c:abs(c['validation_psnr_db']-target)) if valid else None
            distinct=len(set(round(c['validation_psnr_db'],5) for c in valid))
            feasible=bool(chosen and abs(chosen['validation_psnr_db']-target)<=args.tolerance)
            if spec.method_kind=='baseline' and distinct<2:
                # Baseline step ignored by underlying implementation or no effective variation.
                feasible=False
            calibrate={'method_id':spec.method_id,'payload_bits':spec.payload_bits,
                       'blindness_tier':spec.blindness_tier,'target_psnr_db':target,
                       'chosen_strength':chosen['strength'] if chosen else None,
                       'validation_psnr_db':chosen['validation_psnr_db'] if chosen else None,
                       'feasible':feasible,'candidate_count':len(candidates),
                       'distinct_validation_operating_points':distinct,
                       'candidates':candidates,'validation_hosts':[p.name for p in train]}
            calibration.append(calibrate)
            print('calibration',spec.method_id,target,'feasible',feasible,flush=True)
            if not feasible:continue
            adapter=configured_adapter(spec,chosen['strength'],params)
            for host_path in test:
                host=load_host_rgb(host_path)
                for wm_path in wm_paths:
                    watermark=load_watermark_binary(wm_path,size=spec.payload_size)
                    run={'method_id':spec.method_id,'host_id':host_path.name,
                         'watermark_id':wm_path.name,'target_psnr_db':target,
                         'chosen_strength':chosen['strength'],'payload_bits':spec.payload_bits,
                         'blindness_tier':spec.blindness_tier,'status':'ok'}
                    try:
                        start=time.perf_counter();image,key,_=adapter.embed(host,watermark,seed=3101)
                        run['embed_seconds']=time.perf_counter()-start
                        run['achieved_psnr_db']=image_quality_metrics(host,image)['psnr_db']
                        run['test_psnr_within_tolerance']=abs(run['achieved_psnr_db']-target)<=args.tolerance
                        decoded,_=adapter.extract(image,key,original_host=host if spec.requires_original_host else None)
                        run.update({f'clean_{k}':v for k,v in watermark_metrics(watermark,decoded).items()})
                        for attack in attacks:
                            r=dict(run,attack_id=attack.attack_id,attack_category=attack.category)
                            try:
                                attacked=apply_attack(image,attack)
                                recovered,_=adapter.extract(attacked,key,original_host=host if spec.requires_original_host else None)
                                r.update({f'attacked_{k}':v for k,v in watermark_metrics(watermark,recovered).items()})
                            except Exception as exc:r.update(status='attack_error',error=f'{type(exc).__name__}: {exc}')
                            test_rows.append(r)
                    except Exception as exc:test_rows.append(dict(run,status='embed_error',error=f'{type(exc).__name__}: {exc}'))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'predeclared_protocol.json').write_text(json.dumps({
        'validation_host_names':[p.name for p in train],'heldout_host_names':[p.name for p in test],
        'validation_seed':2026,'test_seed':3101,'targets_db':args.targets,'tolerance_db':args.tolerance,
        'attack_suite':args.suite,'attack_ids':[a.attack_id for a in attacks],
        'warning':'Only strength-tunable baselines with equal payload and achieved PSNR within tolerance are fair matched comparisons; others are reported infeasible, not coerced'},indent=2))
    dump(args.output,'calibration',calibration,{'splits':'host-disjoint','all_failures_retained':True})
    dump(args.output,'heldout_matched_results',test_rows,{'splits':'host-disjoint','all_failures_retained':True})
    print('Saved',args.output,'test_rows',len(test_rows),flush=True)
if __name__=='__main__':main()
