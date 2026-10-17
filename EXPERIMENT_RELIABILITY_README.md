# DCT–QR experiment reliability upgrade (experimental layer only)

This patch is relative to the uploaded `dct_qr-main.zip` (17 registered methods: the proposed `dct_qr_theory` and 16 existing baselines). It does **not** change any proposal mathematics, baseline algorithms, attacks, training data, or watermark bits. The default full protocol remains 13 host files × two 64×64 watermarks × three seeds × 37 reviewer-strength attacks × 17 methods = **1,326 embedding trials** and up to **49,062 attack records**. One Kaggle session may not be sufficient.

## Files to copy to GitHub

Copy the contents of this patch ZIP into the repository root, keeping the exact relative paths, then commit and push. The ZIP contains only new/modified files, **not** the images, existing methods, or historical results.

## Improvements

- `scripts/run_unified_benchmark.py`: atomic trial-level saving; `--max-seconds`, `--max-trials`, and `--retry-failed`; stop after the current trial on SIGTERM/SIGINT when possible; never repeatedly retry saved failed trials by default; check source fingerprint and trial identity; reject incomplete, duplicated, missing or non-finite attack metrics. Writes `progress.json`, `run_summary.json`, and aggregates at each session exit. A failed or partial run is never labeled complete.
- `scripts/audit_benchmark.py`: exact expected method × host × watermark × seed × attack coverage; validates outputs, source/input hashes, stale files and numerical metrics; produces `audit.json` and `per_method_coverage.csv`. Use `--strict` when publication-ready completeness is required.
- `scripts/run_reviewer_report.py`: confidence intervals and comparisons now use only strictly validated complete trials, not partial or failed trials.
- `scripts/run_reviewer_diagnostics.py`: atomic, fingerprint-bound per-combination checkpoint chunks for ablation, coupled sensitivity, security controls, and wall-clock timing. Run the same command again to resume. Already observed errors are preserved honestly. `edge` and `gallery` are short and still write completed outputs conventionally.
- `scripts/run_matched_psnr.py`: resume independently calibrated method–PSNR-target chunks; reports both validation feasibility and held-out actual PSNR tolerance. No infeasible target is passed off as a fair comparison.
- `scripts/run_kaggle_experiments.py`: staged runner, automatic periodic ZIP snapshots, reproducible environment variables, safe ZIP restore, final snapshot, and status log.
- `tests/`: expanded integrity, metrics-completeness and checkpoint validation tests.

## Kaggle

Set **Internet ON**, CPU works. Paste `KAGGLE_ONE_CELL.py` into one notebook Python cell after pushing the patch to your GitHub main branch. The default stage is the complete reviewer-strength benchmark. On the next notebook session, upload `dct_qr_checkpoint.zip` to a Kaggle dataset and set `RESTORE_ZIP` to its **real** location under `/kaggle/input`. The checkpoint archive is saved to `/kaggle/working/dct_qr_checkpoint.zip` at session completion and periodically during execution. **Keep the source commit and protocol identical** when restoring. If the commit changes scientific inputs, begin a fresh result series.

CLI alternatives from repo root:

```bash
python -m pip install -e .
python -m pytest -q tests
python scripts/run_kaggle_experiments.py --stage smoke --profile smoke --session-minutes 30
python scripts/run_kaggle_experiments.py --stage benchmark --session-minutes 660
python scripts/run_kaggle_experiments.py --stage diagnostic --diagnostic-mode ablation --profile full
python scripts/run_kaggle_experiments.py --stage diagnostic --diagnostic-mode security --profile full
python scripts/run_kaggle_experiments.py --stage diagnostic --diagnostic-mode sensitivity --profile full
python scripts/run_kaggle_experiments.py --stage diagnostic --diagnostic-mode edge --profile full
python scripts/run_kaggle_experiments.py --stage diagnostic --diagnostic-mode timing --profile full
python scripts/run_kaggle_experiments.py --stage matched --profile full
```

Each reviewer diagnostic or matched stage is separate, allowing the most useful studies to be run without discarding completed core benchmark trials. Run multiple Kaggle sessions as necessary; do not expect all stages to finish in one session.

## Validation and limitations

Executed locally on original repo plus this patch:
- `pytest`: **16/16 passed**.
- Two-method, two-watermark, two-attack benchmark smoke: **4/4 trials and 8/8 attack rows passed**, strict `audit.json` complete, ZIP integrity checked.
- Matched-PSNR one-target calibration and resume: passed; output correctly distinguishes validation and held-out feasibility.
- Diagnostic `security`, `ablation`, `sensitivity`, `timing`, `edge` modes each executed with a small input subset. The security mode's second invocation resumed saved chunks.
- An additional full-method smoke session was interrupted after **19/34** trials; its run summary correctly reported **15 missing** rather than incorrectly claiming success. Thus **17/17 methods were not independently verified in this new session**.

The old extended checkpoint has a different code fingerprint because the runner and experiment scripts have changed. Keep it as historical pre-patch evidence, do not combine it with new results. The post-patch full 1,326-trial benchmark has **not** been executed here.

**Known algorithmic limitations remain**: all-black, all-gray, and all-white embedding may raise a certificate numerical error; the saturated checkerboard can have poor clean BER. These are accurately surfaced by `--diagnostic-mode edge` and are NOT automatically passed. Fixing these requires separate proposed-method changes and revised theoretical/experimental validation. The current patch concentrates on experiment integrity and reliable long runs.

**Publication**: report only strict, complete results from one fingerprint/commit. The 37-attack protocol, validation-only PSNR calibration, negative controls and video/slideshow throughput are different experimental settings and must not be silently pooled with the manuscript's original 20×3 benchmark.
