#!/usr/bin/env python3
"""Convenience orchestrator for reproducible reviewer analyses.

No auto-install, network access, or changes to proposal/baseline code.
Use --profile smoke for a short integration check or --profile full for the
full 13-host/two-watermark/16-existing-baseline evaluation (CPU intensive).
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(command,env,log):
    print('RUN:', ' '.join(command),flush=True)
    record={'command':command,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    started=time.perf_counter()
    proc=subprocess.run(command,cwd=ROOT,env=env)
    record['exit_code']=proc.returncode
    record['elapsed_seconds']=round(time.perf_counter()-started,4)
    log.append(record)
    if proc.returncode!=0:raise RuntimeError(f'Failed: {" ".join(command)}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['smoke','full'],default='smoke')
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewer_session')
    p.add_argument('--zip-results',action='store_true')
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('JILP_NUM_THREADS','1')
    py=sys.executable
    manifest=[]
    try:
        if args.profile=='smoke':
            protocol='configs/reviewer/smoke.json';suite='sanity';limit=['--partition','all','--host-limit','1','--watermark-limit','1','--attack-limit','2']
        else:
            protocol='configs/reviewer/reviewer_strength.json';suite='reviewer_strength';limit=['--partition','test']
        run([py,'scripts/run_unified_benchmark.py','--protocol',protocol,'--output-dir',str(out/'benchmarks'),'--resume'],env,manifest)
        proto=json.loads((ROOT/protocol).read_text())
        bench=out/'benchmarks'/proto['protocol_id']
        run([py,'scripts/run_reviewer_report.py','--run-dir',str(bench)],env,manifest)
        for mode in ('ablation','security','edge','timing','gallery'):
            run([py,'scripts/run_reviewer_diagnostics.py','--mode',mode,'--output',str(out/mode),
                  '--attack-suite',suite,*limit],env,manifest)
        run([py,'scripts/run_comparison_gallery.py','--output',str(out/'comparison_gallery'),
             '--suite',suite,'--attack-limit','2' if args.profile=='smoke' else '8'],env,manifest)
        run([py,'scripts/run_frame_stream_benchmark.py','--output',str(out/'frame_stream'),
             '--max-frames','2' if args.profile=='smoke' else '12'],env,manifest)
        if args.profile=='full':
            run([py,'scripts/run_reviewer_diagnostics.py','--mode','sensitivity','--partition','validation',
                '--output',str(out/'sensitivity'),'--attack-suite','common20'],env,manifest)
            run([py,'scripts/run_matched_psnr.py','--output',str(out/'matched_psnr'),
                '--suite','common20'],env,manifest)
    finally:
        (out/'execution_log.json').write_text(json.dumps({'profile':args.profile,'commands':manifest},indent=2))
    if args.zip_results:
        archive=shutil.make_archive(str(out),'zip',root_dir=out)
        print('RESULT ARCHIVE:',archive,flush=True)
    print('COMPLETE:',out,flush=True)
if __name__=='__main__':main()
