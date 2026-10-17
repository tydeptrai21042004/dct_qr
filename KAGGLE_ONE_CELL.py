# Paste this entire file into ONE Kaggle Python notebook cell.
# Settings: Internet ON; CPU is sufficient; save checkpoints from Output.
# Upload a prior dct_qr_checkpoint.zip as a Kaggle Dataset for later sessions,
# then set RESTORE_ZIP to its exact /kaggle/input/... path.
import os, sys, subprocess
from pathlib import Path

REPOSITORY = 'https://github.com/tydeptrai21042004/dct_qr.git'
BRANCH = 'main'
RESTORE_ZIP = None  # e.g. '/kaggle/input/previous-dctqr/dct_qr_checkpoint.zip'
SESSION_MINUTES = 650  # stay below the notebook session limit
STAGE = 'benchmark'  # 'smoke', 'benchmark', 'diagnostic', or 'matched'
DIAGNOSTIC_MODE = 'ablation'  # also sensitivity, security, edge, timing, gallery
RETRY_FAILED = False  # True only for deliberate reattempts of failed trials

work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
repo = work/'dct_qr'
if not (repo/'.git').exists():
    subprocess.run(['git','clone','--depth','1','--branch',BRANCH,REPOSITORY,str(repo)],check=True)
else:
    print('Using existing clone:',repo)
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(repo)],check=True)
env = os.environ.copy()
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JILP_NUM_THREADS','NUMBA_NUM_THREADS'):
    env[variable]='1'
subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=repo,env=env,check=True)
command=[sys.executable,'scripts/run_kaggle_experiments.py',
         '--stage',STAGE,'--session-minutes',str(SESSION_MINUTES),
         '--snapshot-minutes','10', '--output',str(work/'dct_qr_results'),
         '--archive',str(work/'dct_qr_checkpoint.zip')]
if RESTORE_ZIP:
    if not Path(RESTORE_ZIP).is_file():raise FileNotFoundError(RESTORE_ZIP)
    command+=['--restore',RESTORE_ZIP]
if STAGE=='diagnostic':command+=['--diagnostic-mode',DIAGNOSTIC_MODE]
if RETRY_FAILED:command.append('--retry-failed')
print('Executing:', ' '.join(command),flush=True)
subprocess.run(command,cwd=repo,env=env,check=True)
print('Checkpoint:',work/'dct_qr_checkpoint.zip')
print('Summary:',work/'dct_qr_results'/'benchmark'/'reviewer_strength_all_existing_methods_v1'/'audit.json')
print('Remember: an interrupted/partial benchmark must NOT be reported as complete.')
