#!/usr/bin/env python3
"""Read a unified benchmark, produce audit-ready reviewer supplementary tables.

No watermark code is changed; no missing rows are imputed. Paired uncertainty
resamples independent host files (cluster bootstrap), NOT attacks or seeds.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.benchmark.aggregation import load_trials,write_csv
from run_unified_benchmark import _trial_issues


def safe_mean(values):
    v=np.asarray([float(x) for x in values if isinstance(x,(float,int)) and np.isfinite(float(x))],dtype=float)
    return float(v.mean()) if len(v) else None


def q(vals,p):
    a=np.asarray(vals,dtype=float)
    return float(np.quantile(a,p)) if a.size else None


def bootstrap_ci(values,seed=2026,reps=2000):
    """Cluster means entered once for each independent host."""
    a=np.asarray(values,dtype=float)
    if len(a)<2:return None,None
    rng=np.random.default_rng(seed)
    reps=int(reps)
    if reps<100: raise ValueError('at least 100 bootstrap replicates required')
    draws=rng.integers(0,len(a),size=(reps,len(a)))
    avg=a[draws].mean(axis=1)
    return q(avg,0.025),q(avg,0.975)


def per_host_key(r):return (r['host_id'],r['watermark_id'],r['seed'],r['attack_id'])


def report(base:Path,out:Path,reference:str,psnr_tolerance:float,reps:int):
    manifest=json.loads((base/'manifest.json').read_text())
    run_summary_path=base/'run_summary.json'
    run_summary=json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
    fingerprint=manifest.get('experiment_fingerprint')
    trials_all=load_trials(base/'trials')
    trials=[t for t in trials_all if t.get('experiment_fingerprint')==fingerprint]
    methods={m['method_id']:m for m in manifest['methods']}
    attack_ids={a['attack_id'] for a in manifest['attacks']}
    host_ids={Path(h['path']).name for h in manifest['hosts']}
    wm_ids={Path(w['path']).name for w in manifest['watermark_files']}
    seeds=set(manifest['protocol']['seeds'])
    expected={(method,host,watermark,int(seed)) for method in methods for host in host_ids for watermark in wm_ids for seed in seeds}
    observed={(t.get('method_id'),t.get('host_id'),t.get('watermark_id'),t.get('seed')) for t in trials}
    duplicates=len(trials)-len(observed)
    failures=[]
    for trial in trials:
        rows=trial.get('rows',[])
        ids={r.get('attack_id') for r in rows}
        fail=[r.get('attack_id') for r in rows if r.get('status')!='ok']
        issues=_trial_issues(trial, protocol_id=manifest['protocol']['protocol_id'],
                             attack_ids=tuple(a['attack_id'] for a in manifest['attacks']),
                             fingerprint=fingerprint)
        if issues:
            failures.append({'method_id':trial.get('method_id'),'host':trial.get('host_id'),
                'watermark':trial.get('watermark_id'),'seed':trial.get('seed'),'trial_status':trial.get('status'),
                'missing_attacks':sorted(attack_ids-ids),'failed_attacks':fail,'validation_issues':issues})
    completeness={
       'experiment_fingerprint':fingerprint,
       'expected_trials':len(expected),'current_trials':len(trials),'stale_trials':len(trials_all)-len(trials),
       'missing_trials':sorted([list(x) for x in expected-observed]),
       'duplicate_trials':duplicates,'failed_or_partial_trials':failures,
       'all_complete':expected==observed and duplicates==0 and not failures}
    out.mkdir(parents=True,exist_ok=True)
    (out/'completeness.json').write_text(json.dumps(completeness,indent=2))

    valid=[]
    valid_trials=[t for t in trials if not _trial_issues(t, protocol_id=manifest['protocol']['protocol_id'],
                      attack_ids=tuple(a['attack_id'] for a in manifest['attacks']), fingerprint=fingerprint)]
    for t in valid_trials:
        for row in t.get('rows',[]):
            if row.get('status')=='ok':valid.append(row)

    per_condition=[]
    groups=defaultdict(list)
    for r in valid:
        groups[(r['method_id'],r['attack_id'])].append(r)
    for (m,atk),rows in sorted(groups.items()):
        hosts=defaultdict(list)
        for r in rows:hosts[r['host_id']].append(float(r['attacked_nc']))
        cluster_means=[safe_mean(vals) for vals in hosts.values()]
        cluster_means=[x for x in cluster_means if x is not None]
        low,high=bootstrap_ci(cluster_means,reps=reps)
        per_condition.append({'method_id':m,'attack_id':atk,
            'attack_category':rows[0]['attack_category'],'attack_severity':rows[0]['attack_severity'],
            'image_n':len(cluster_means),'trial_n':len(rows),'host_cluster_mean_nc':safe_mean(cluster_means),
            'mean_ber':safe_mean([r.get('attacked_ber') for r in rows]),
            'mean_psnr_db':safe_mean([r.get('embedding_psnr_db') for r in rows]),
            'mean_extract_ms':safe_mean([float(r['extract_seconds'])*1e3 for r in rows if r.get('extract_seconds') is not None]),
            'nc_95ci_low':low,'nc_95ci_high':high,'min_nc':min(float(r['attacked_nc']) for r in rows),
            'nc_q05':q([r['attacked_nc'] for r in rows],0.05)})
    write_csv(out/'per_attack_ci.csv',per_condition)

    # Strictly paired: same host, watermark, embedding seed, attack ID. No
    # comparisons where capacity differs; PSNR mismatch is reported, not hidden.
    by_method=defaultdict(dict)
    for r in valid:by_method[r['method_id']][per_host_key(r)]=r
    paired=[]
    ref_rows=by_method.get(reference,{})
    for method, rows in sorted(by_method.items()):
        if method==reference:continue
        common=sorted(ref_rows.keys() & rows.keys())
        groups=defaultdict(list)
        for key in common:
            proposal,baseline=ref_rows[key],rows[key]
            if int(proposal['payload_bits'])!=int(baseline['payload_bits']):continue
            psnr_gap=abs(float(proposal['embedding_psnr_db'])-float(baseline['embedding_psnr_db']))
            groups[(baseline['attack_id'],psnr_gap<=psnr_tolerance)].append((proposal,baseline,psnr_gap))
        for (attack_id,matched),pairs in sorted(groups.items()):
            by_host=defaultdict(list)
            for prop,base,gap in pairs:
                by_host[prop['host_id']].append(float(prop['attacked_nc'])-float(base['attacked_nc']))
            differences=[safe_mean(v) for v in by_host.values()]
            differences=[v for v in differences if v is not None]
            low,high=bootstrap_ci(differences,reps=reps)
            paired.append({'method_id':method,'reference':reference,'attack_id':attack_id,
                'capacity_matched':True,'PSNR_within_tolerance':matched,'psnr_tolerance_db':psnr_tolerance,
                'note':'observed PSNR stratum, not tuned matched-distortion experiment',
                'n_pairs':len(pairs),'n_independent_hosts':len(differences),
                'mean_psnr_gap_db':safe_mean([x[2] for x in pairs]),
                'mean_nc_gain':safe_mean(differences),'paired_95ci_low':low,'paired_95ci_high':high,
                'proposal_blindness':methods[reference]['blindness_tier'],
                'baseline_blindness':methods[method]['blindness_tier']})
    write_csv(out/'paired_capacity_psnr_strata.csv',paired)
    # Honest runtime and serialized key disclosure in one place.
    timing=[]
    for m in methods:
        mt=[t for t in valid_trials if t.get('method_id')==m]
        extract=[float(t['clean_extract_seconds']) for t in mt if isinstance(t.get('clean_extract_seconds'),(int,float))]
        embed=[float(t['embed_seconds']) for t in mt if isinstance(t.get('embed_seconds'),(int,float))]
        keys=[float(t['key_size_bytes']) for t in mt if isinstance(t.get('key_size_bytes'),(int,float))]
        spec=methods[m]
        timing.append({'method_id':m,'blindness_tier':spec['blindness_tier'],
              'requires_original_host':spec['requires_original_host'],
              'payload_bits':spec['payload_size']**2,'complete_clean_trials':len(mt),
              'embed_mean_ms':safe_mean([x*1000 for x in embed]),'embed_p95_ms':q([x*1000 for x in embed],0.95),
              'extract_mean_ms':safe_mean([x*1000 for x in extract]),'extract_p95_ms':q([x*1000 for x in extract],0.95),
              'serialized_key_mean_bytes':safe_mean(keys),'serialized_key_p95_bytes':q(keys,0.95)})
    write_csv(out/'latency_key_size.csv',timing)
    (out/'report_metadata.json').write_text(json.dumps({
        'is_complete':completeness['all_complete'], 'uncertainty_unit':'host filename (cluster bootstrap)',
        'bootstrap_reps':reps,'baseline_matching':'identical host+watermark+seed+attack and equal payload bits',
        'psnr_strata':'observed PSNR only; NOT equivalent to independently tuned matched-PSNR design',
        'missing_or_error_rows':'excluded from metrics but reported explicitly in completeness.json',
        'all_methods':list(methods),'attack_count':len(attack_ids),'reference_method':reference},indent=2))
    print(json.dumps({'out':str(out),'complete':completeness['all_complete'],
                      'per_attack_rows':len(per_condition),'paired_rows':len(paired)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reference',default='dct_qr_theory')
    parser.add_argument('--psnr-tolerance',type=float,default=0.5)
    parser.add_argument('--bootstrap-reps',type=int,default=2000)
    args=parser.parse_args()
    report(args.run_dir,args.output_dir or args.run_dir/'reviewer_report',args.reference,args.psnr_tolerance,args.bootstrap_reps)

if __name__=='__main__':main()
