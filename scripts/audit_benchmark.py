#!/usr/bin/env python3
"""Strict experiment matrix audit. Never substitute missing or failed trials.

Run only on a benchmark directory containing manifest.json and trials/*.json.
Produces audit.json and per_method.csv; --strict fails if anything is missing,
invalid or belongs to a different experiment fingerprint.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_unified_benchmark import _trial_issues


def _hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def audit(base: Path, verify_sources: bool = True) -> dict:
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    proto=manifest['protocol']; fp=manifest['experiment_fingerprint']
    attacks=tuple(a['attack_id'] for a in manifest['attacks'])
    expected={(m['method_id'],Path(h['path']).name,Path(w['path']).name,int(seed))
              for m in manifest['methods'] for h in manifest['hosts']
              for w in manifest['watermark_files'] for seed in proto['seeds']}
    records={}; unrecognized=[]; invalid=[]
    for path in sorted((base/'trials').glob('*.json')):
        try: trial=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc:
            invalid.append({'file':path.name,'issues':['unparseable_json',type(exc).__name__]})
            continue
        if not isinstance(trial,dict):
            invalid.append({'file':path.name,'issues':['not_object']});continue
        key=(trial.get('method_id'),trial.get('host_id'),trial.get('watermark_id'),trial.get('seed'))
        if trial.get('experiment_fingerprint')!=fp or key not in expected:
            unrecognized.append(path.name);continue
        if key in records:
            invalid.append({'file':path.name,'issues':['duplicate_trial_identity']})
        issues=_trial_issues(trial,protocol_id=proto['protocol_id'],attack_ids=attacks,
                            fingerprint=fp,identity=key)
        records[key]=issues
        if issues:invalid.append({'file':path.name,'identity':list(key),'issues':issues})
    missing=sorted([list(k) for k in expected-records.keys()])
    per_method=[]
    for m in manifest['methods']:
        mid=m['method_id']; keys=[k for k in expected if k[0]==mid]
        completed=sum(k in records and not records[k] for k in keys)
        failures=sum(k in records and bool(records[k]) for k in keys)
        per_method.append({'method_id':mid,'expected':len(keys),'passed':completed,
                           'failed':failures,'missing':len(keys)-completed-failures,
                           'is_complete':completed==len(keys)})
    source_mismatch=[]
    if verify_sources:
        for rel,digest in {**manifest.get('source_sha256',{}),
                           **{v['path']:v['sha256'] for v in manifest['hosts']},
                           **{v['path']:v['sha256'] for v in manifest['watermark_files']}}.items():
            p=ROOT/rel
            if not p.is_file() or _hash(p)!=digest: source_mismatch.append(rel)
        params=ROOT/'configs/benchmark/baseline_parameters.json'
        if _hash(params)!=manifest.get('baseline_parameters_sha256'):
            source_mismatch.append(str(params.relative_to(ROOT)))
    n_valid=sum(not v for v in records.values())
    return {'protocol_id':proto['protocol_id'],'fingerprint':fp,'expected_trials':len(expected),
            'complete_valid_trials':n_valid,'failed_or_invalid_trial_count':len(invalid),
            'missing_trial_count':len(missing),'stale_or_unrecognized_count':len(unrecognized),
            'source_or_input_mismatch_count':len(source_mismatch),
            'attack_count_per_complete_trial':len(attacks),
            'all_complete':(n_valid==len(expected) and not invalid and not unrecognized and not source_mismatch),
            'per_method':per_method,'missing_trials':missing,'invalid_trials':invalid,
            'stale_or_unrecognized_files':unrecognized,'source_or_input_mismatches':source_mismatch,
            'warning':'Only complete valid trials are eligible for fully supported aggregate claims. '
                      'Attack success is not proof of scientific method correctness.'}


def write_audit(base: Path, *, strict: bool=False, verify_sources: bool=True) -> dict:
    data=audit(base,verify_sources=verify_sources)
    (base/'audit.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (base/'per_method_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=('method_id','expected','passed','failed','missing','is_complete'))
        wr.writeheader();wr.writerows(data['per_method'])
    print(json.dumps({k:v for k,v in data.items() if k not in
          ('missing_trials','invalid_trials','per_method','stale_or_unrecognized_files')},indent=2),flush=True)
    if strict and not data['all_complete']:
        raise SystemExit('Benchmark incomplete: see audit.json (nonzero exit by design).')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--strict',action='store_true')
    p.add_argument('--no-verify-sources',action='store_true')
    a=p.parse_args()
    write_audit(a.run_dir,strict=a.strict,verify_sources=not a.no_verify_sources)
