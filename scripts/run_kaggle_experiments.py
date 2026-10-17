#!/usr/bin/env python3
"""Reproducible CPU Kaggle session runner with periodic crash-resistant archives.

Stages:
  smoke       tests + two attacks on all registered methods + strict audit
  benchmark   13 hosts x two watermarks x three seeds x 37 attacks x all methods
  diagnostic  one pre-existing reviewer diagnostic mode (independent evidence)
  matched     validation-only calibration + held-out matched-PSNR experiment

No proposal or baseline implementation is modified. Full benchmark can take
multiple sessions; partial results are explicitly marked incomplete.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=Path('/kaggle/working/dct_qr_results') if Path('/kaggle/working').exists() else ROOT/'results'/'kaggle_sessions'


def _snapshot(output:Path, archive:Path) -> None:
    """Write into a temporary archive, then atomically replace the last snapshot."""
    archive.parent.mkdir(parents=True,exist_ok=True)
    tmp=archive.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith('.tmp') and not path.name.endswith('.partial'):
                    z.write(path,arcname='dct_qr_results/'+str(path.relative_to(output)))
        os.replace(tmp,archive)
    finally:
        if tmp.exists():tmp.unlink()


def restore(archive:Path,output:Path) -> None:
    if not archive.is_file():raise FileNotFoundError(archive)
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'Checkpoint ZIP corrupted at {bad}')
        for info in z.infolist():
            name=Path(info.filename)
            parts=name.parts
            if not parts or parts[0]!='dct_qr_results' or '..' in parts or name.is_absolute():
                raise ValueError(f'Unsafe or unexpected checkpoint entry: {info.filename}')
            path=output.joinpath(*parts[1:])
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError(f'Unsafe checkpoint destination: {path}')
            if info.is_dir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            # Reject overwrite of conflicting old/new versions to avoid mixed evidence.
            data=z.read(info)
            if path.exists() and path.read_bytes()!=data:
                raise ValueError(f'Checkpoint conflicts with existing output: {path}. Use an empty output directory.')
            path.write_bytes(data)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','benchmark','diagnostic','matched'),default='benchmark')
    p.add_argument('--diagnostic-mode',choices=('ablation','sensitivity','security','edge','timing','gallery'),default='ablation')
    p.add_argument('--profile',choices=('smoke','full'),default='full')
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--archive',type=Path)
    p.add_argument('--restore',type=Path,help='Previously downloaded dct_qr_checkpoint.zip')
    p.add_argument('--session-minutes',type=float,default=660,
                   help='Budget including final export; default 11 hours for Kaggle.')
    p.add_argument('--snapshot-minutes',type=float,default=10)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--methods',default='all',help='Existing method selector for benchmark/smoke.')
    p.add_argument('--test-host-limit',type=int,default=1)
    p.add_argument('--max-trials',type=int,help='Useful for local smoke/recovery tests.')
    p.add_argument('--diagnostic-partition',choices=('validation','test','all'),default='test')
    p.add_argument('--diagnostic-host-limit',type=int)
    p.add_argument('--diagnostic-attack-limit',type=int)
    a=p.parse_args()
    if a.session_minutes<=0 or a.snapshot_minutes<=0:p.error('Budgets must be positive')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    archive=(a.archive or output.parent/'dct_qr_checkpoint.zip').resolve()
    if archive.is_relative_to(output):p.error('--archive must be OUTSIDE --output to avoid recursive ZIPs')
    if a.restore:restore(a.restore.resolve(),output)
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JILP_NUM_THREADS','NUMBA_NUM_THREADS'):
        env[name]='1'
    planned=min(a.session_minutes*60-90, a.session_minutes*60*0.90)
    if planned<=0:p.error('Session too short to reserve time for archive')
    py=sys.executable
    if a.stage=='smoke':
        cmds=[[py,'-m','pytest','-q','tests'],
            [py,'scripts/run_unified_benchmark.py','--protocol','configs/reviewer/reviewer_strength.json',
             '--methods',a.methods,'--host-limit',str(a.test_host_limit),'--attack-limit','2',
             '--seeds','2026','--output-dir',str(output/'benchmark_smoke'),'--resume']]
        proto='reviewer_strength_all_existing_methods_v1';base=output/'benchmark_smoke'/proto
    elif a.stage=='benchmark':
        proto='reviewer_strength_all_existing_methods_v1';base=output/'benchmark'/proto
        cmds=[[py,'scripts/run_unified_benchmark.py','--protocol','configs/reviewer/reviewer_strength.json',
               '--methods',a.methods,'--output-dir',str(output/'benchmark'),'--resume',
               '--max-seconds',str(max(1,planned))]]
    elif a.stage=='diagnostic':
        mode=a.diagnostic_mode
        flags=['--mode',mode,'--partition',a.diagnostic_partition,
               '--attack-suite','reviewer_strength' if a.profile=='full' else 'sanity',
               '--output',str(output/'diagnostics'/mode)]
        if a.diagnostic_host_limit:flags+=['--host-limit',str(a.diagnostic_host_limit)]
        if a.diagnostic_attack_limit:flags+=['--attack-limit',str(a.diagnostic_attack_limit)]
        if a.profile=='smoke':flags+=['--host-limit','1','--watermark-limit','1','--attack-limit','2']
        cmds=[[py,'scripts/run_reviewer_diagnostics.py',*flags]]
        base=None
    else:
        args=['--output',str(output/'matched_psnr'), '--suite','common20',
              '--methods',a.methods if a.methods!='all' else 'common_4096']
        if a.profile=='smoke':
            args+=['--targets','54','--validation-limit','1','--test-limit','1',
                   '--attack-limit','2','--strengths','4','8']
        cmds=[[py,'scripts/run_matched_psnr.py',*args]]
        base=None
    if a.max_trials and a.stage in ('benchmark','smoke'):
        cmds[-1]+=['--max-trials',str(a.max_trials)]
    if a.retry_failed and a.stage in ('benchmark','smoke'):cmds[-1].append('--retry-failed')

    stop=threading.Event()
    def snapshots():
        while not stop.wait(a.snapshot_minutes*60):
            try:
                _snapshot(output,archive)
                print('PERIODIC CHECKPOINT:',archive,flush=True)
            except Exception as exc:
                print('CHECKPOINT ERROR:',type(exc).__name__,exc,flush=True)
    background=threading.Thread(target=snapshots,daemon=True);background.start()
    commands=[];failed=False
    start=time.perf_counter()
    try:
        for cmd in cmds:
            print('RUN:', ' '.join(cmd),flush=True)
            current={'command':cmd,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            done=subprocess.run(cmd,cwd=ROOT,env=env,check=False)
            current.update(exit_code=done.returncode,elapsed_seconds=time.perf_counter()-start)
            commands.append(current)
            if done.returncode!=0:
                failed=True
                break
            if time.perf_counter()-start > planned:
                print('Time budget reached; stop before next stage.',flush=True)
                break
        if base is not None and (base/'manifest.json').exists():
            check=subprocess.run([py,'scripts/audit_benchmark.py','--run-dir',str(base)],cwd=ROOT,env=env)
            if check.returncode:failed=True
            if (base/'manifest.json').exists():
                report=subprocess.run([py,'scripts/run_reviewer_report.py','--run-dir',str(base)],cwd=ROOT,env=env)
                if report.returncode:failed=True
        overall={'stage':a.stage,'profile':a.profile,'commands':commands,
                 'process_failures':failed,'stage_completed_without_process_errors':not failed,
                 'scientific_completeness_must_be_read_from':'benchmark/<protocol>/audit.json',
                 'elapsed_seconds':time.perf_counter()-start}
        (output/'session_log.json').write_text(json.dumps(overall,indent=2))
    finally:
        stop.set();background.join(timeout=2)
        _snapshot(output,archive)
        print('FINAL CHECKPOINT:',archive,flush=True)
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
