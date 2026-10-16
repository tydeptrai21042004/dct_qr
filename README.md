# Corrected DCT-QR Theory + Baselines (Minimal Runnable Project)

This is a **source-only slimmed extraction** of `Hanoi-main(20261016-153526).zip`.
It contains the **corrected `dct_qr_theory` proposed method, not the older `dct_qr`**,
all **16 existing baseline implementations**, the shared attacks and metrics,
all 13 input BMP hosts, and both input PNG watermarks. Other proposals, old
result folders, diagrams, reports, and unrelated scripts are absent.

## Run (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_unified_benchmark.py --protocol configs/benchmark/quick.json
```

For a focused 1-image/3-attack check (corrected method + one baseline):

```bash
python scripts/run_unified_benchmark.py --protocol configs/benchmark/quick.json \
  --methods dct_qr_theory,dct_dm_qim_chen2001_blind --host-limit 1 --attack-limit 3
```

Complete **13 host × 2 watermarks × all 17 methods** run with the full attack suite:

```bash
python scripts/run_unified_benchmark.py --protocol configs/benchmark/publication.json --resume
```

For the balanced common-20 suite across all available inputs, run:

```bash
python scripts/run_unified_benchmark.py --protocol configs/benchmark/publication.json \
  --suite common20 --seeds 2026 --resume
```

The original publication configuration uses three seeds and a much larger
publication attack set; completion may be computationally intensive. The
benchmark writes atomic JSON trials and CSV/JSON aggregates to
`results/theory_baselines/`. Interrupted runs can be resumed.

One-factor experiments (one host):

```bash
python scripts/run_theory_ablation.py --attack-suite common20
python scripts/run_dct_qr_theory_numerics.py
```

## Scientific scope and warnings

- Only proposed method: `dct_qr_theory`. The public method registry **rejects** `dct_qr`.
- `src/qr64_certified/proposals/qr_carrier.py` consists of two unchanged
  common carrier/Arnold helpers extracted from the legacy implementation; it
  is **not** the old proposed algorithm.
- Extraction for the corrected proposal does not use the original host image,
  **but does use host-derived parameters in its key**. Call it *host-blind,
  key-assisted*, not strictly blind.
- The 16 baselines have mixed blind/semi-blind/non-blind assumptions and some
  use **16×16 watermark payloads instead of 64×64**. Interpret group comparisons
  according to `blindness_tier`, `comparison_group`, `payload_bits`, and the
  scientific adaptation disclosures in `src/qr64_certified/baselines/metadata.py`.
- The baseline code is retained rather than re-implemented; its original math
  and any existing paper-adaptation limitations are unchanged.
- Results are computed **at runtime**, not pre-filled from earlier experiment
  reports. Numerical performance on different hosts or attacks is not guaranteed.

---

# Reviewer experiment expansion — no changes to the method or baselines

**Experimental-only additions:** all algorithm files in `src/qr64_certified/proposals/`,
`src/qr64_certified/_jilp_core/` and `src/qr64_certified/baselines/` are unchanged.
There is still exactly **one proposed method** (`dct_qr_theory`) and **16 pre-existing
baselines**. New attack operators, experiment scripts, source fingerprints, and
result-processing utilities are the only changes.

## Start here

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_reviewer_suite.py --profile smoke --zip-results
```

For the **full reviewer suite** (potentially very large CPU workload):

```bash
python scripts/run_reviewer_suite.py --profile full --zip-results
```

The full profile uses all supplied input BMP images and both watermarks for the
method-and-baseline comparison, and it runs the additional analyses on a
predeclared held-out host split. Results stay under `results/` and include a
session execution log. All successful runs are reproducible with fixed seeds;
performance is not guaranteed.

## Individual experiment groups

| Reviewer requirement | Command / output |
|---|---|
| Strong JPEG/noise/geometry, off-grid rotations, shears, fractional shifts, targeted DCT-band attenuation | `python scripts/run_unified_benchmark.py --protocol configs/reviewer/reviewer_strength.json --resume` |
| Independent holdout with identical host, attack and payload bookkeeping | `python scripts/run_reviewer_diagnostics.py --mode ablation --partition test --attack-suite reviewer_strength` |
| Full + six single-factor + two double-factor ablations (attack by attack) | `python scripts/run_reviewer_diagnostics.py --mode ablation --partition test` |
| Coupled reference-step × pilot count × pilot-step 27-point sensitivity | `python scripts/run_reviewer_diagnostics.py --mode sensitivity --partition validation --attack-suite common20` |
| Wrong-seed valid key, unwatermarked host, empirical FPR/TPR thresholds | `python scripts/run_reviewer_diagnostics.py --mode security --partition test` |
| Extreme dark/bright, flat and saturated synthetic images | `python scripts/run_reviewer_diagnostics.py --mode edge` |
| Repeated wall-clock embed + clean extract, excluding warmups | `python scripts/run_reviewer_diagnostics.py --mode timing --partition test --methods all` |
| Extracted watermark comparison gallery (first selected method) | `python scripts/run_reviewer_diagnostics.py --mode gallery --partition test --methods dct_qr_theory --attack-suite reviewer_strength` |
| Independently calibrated matched-PSNR, equal-capacity subset | `python scripts/run_matched_psnr.py --methods common_4096 --targets 50 52 54` |
| Paired host-cluster confidence intervals, all per-attack rows, honest matched-PSNR *strata*, completeness | `python scripts/run_reviewer_report.py --run-dir results/reviewer/reviewer_strength_all_existing_methods_v1` |

Run smaller targeted checks by adding `--host-limit 1 --watermark-limit 1
--attack-limit 2` where the selected script supports these options.

## Experimental and statistical rules

1. **Host holdout:** `run_reviewer_diagnostics.py` and `run_matched_psnr.py`
   use a fixed SHA-256 host-filename split (default 9 validation / 4 test images).
   All watermarks and attack variants of one host remain in the same split.
   This is not a validation of generalization on a large external dataset.
2. **Units:** Gaussian `sigma` uses RGB intensity units 0–255; normalized
   Gaussian and speckle variance are explicitly named as variance on [0,1].
   Rotation tests labeled `rotation_back` involve **two** resampling steps and
   are not the same as uncorrected rotation.
3. **No cherry-picking:** failed embeds, missing attacks, mismatched payload
   sizes, nonmatched PSNR, and uncalibratable baselines are recorded, not assigned
   successful scores. `completeness.json` must report `all_complete: true`
   before treating a set of results as exhaustive.
4. **Statistics:** bootstrap CIs resample **host** IDs rather than treating
   attacks, watermark variants or repeated seeds as independent samples. With
   only four held-out hosts, intervals have substantial uncertainty.
5. **Matched PSNR:** `run_matched_psnr.py` tunes only the configurable strength
   exposed by each existing implementation on validation hosts. Baselines that
   cannot achieve a requested PSNR are **infeasible**. It does not alter
   algorithm equations. The simpler report's `paired_capacity_psnr_strata.csv`
   is observational and explicitly **not** a calibrated comparison.
6. **Timing:** serial still-image `1/(embed+extract)` throughput is not
   equivalent to live video throughput or a guaranteed frame-deadline pass.
   All timing figures are hardware- and process-dependent, with Python/JIT
   warmup effects disclosed.
7. **Security:** negative-control detection is based on the similarity statistic
   `1 − BER`, not authenticated cryptographic detection. This is not a proof
   of encryption, password entropy, or adversarial key security.
8. **Side information:** host-dependent QR and scheduling parameters remain
   necessary; the method is *host-blind, key-assisted*, not strict blind.
   The serialization cost is recorded in both the main benchmark and reports.
9. **Fixed mathematical method:** capacity stays **4096 bits on 512×512** with
   8×8 blocks. Arbitrarily changing payload count, block dimensions, RGB
   embedding channel or adding temporal voting/ECC would require changing
   the proposed method and is deliberately not done. Consequently, those
   reviewer suggestions are reported as **out of scope**, not simulated as
   completed scientific experiments.
10. **Limits:** neither trained deep-learning watermark removers nor genuine
    live-video tests are included; those require external implementations or
    real video data and are not misrepresented as supported.

## Traceability

`manifest.json` contains SHA-256 hashes of inputs, source files, configuration
and parameters. Resume skips a trial **only** if its fingerprint matches, its
status is `ok`, every expected attack appears, and no attack is marked failed.
Output folders contain JSON and CSV raw evidence and, for the gallery, PNG
figures. Results are generated on execution, not pre-filled or hard-coded.

## Optional independent frame-stream and side-by-side galleries

```bash
# Synthetic slideshow from included BMP images; NOT a real-video experiment.
python scripts/run_frame_stream_benchmark.py --max-frames 12 --codec-quality 70

# Real video supplied separately by the investigator, with each frame
# center-cropped/resized to 512x512 before embedding:
python scripts/run_frame_stream_benchmark.py --video /path/to/clip.mp4 --max-frames 100

# Side-by-side extracted watermark panels; both methods already exist.
python scripts/run_comparison_gallery.py \
  --methods dct_qr_theory,dct_dm_qim_chen2001_blind \
  --suite reviewer_strength --attack-limit 8
```

The stream timing result measures **serial per-frame processing only**, with
real preprocessing and JPEG channel costs. It is not a live-stream or
steady-state video watermarking proof. The per-frame key may change by frame;
the watermark method is unchanged, and no temporal voting is added.

`run_reviewer_diagnostics.py --mode security` also tests two **tampered
side-information keys** (reversed payload index ordering and flipped pilot
bits). Failures are retained in JSON/CSV; these controls do not constitute
cryptographic authentication or a proof of key security.
