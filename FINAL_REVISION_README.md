# FINAL DCT–QR reviewer experiment patch (2026-11-10)

**Apply to the ORIGINAL `dct_qr-main (1).zip` project** (one consolidated patch,
not on top of the earlier patches). Only changed/new files are distributed.
The proposed `src/qr64_certified/proposals/` algorithms, embedding, extraction,
DCT coefficients, QR scale compensation, and all theoretical constants are
unchanged. No Lena/Lenna host can be used. No precomputed attack results are
shipped; missing Table 10 rows must not be replaced by zero.

## Exactly what differs from the supplied manuscript / supervisor requests

* The paper's original Table 3 still specifies Gaussian sigma 1/3/5,
  rotation **and inverse** 0.5/1/2 degrees, and crop-and-resize. The NEW
  canonical revision protocol uses Gaussian **variance 0.001/0.003/0.005**
  on [0,1] pixels, rotation **once** 5/10/15 degrees on a fixed-size canvas,
  and **no-resize center crop on a zero-filled 512x512 canvas** with keep
  0.95/0.90/0.80. Each change REQUIRES re-running all methods and replacing
  every old aggregate and attack-dependent statement in the manuscript.
* Alternate sigma-only suite available for reproducing the old Gaussian
  convention, but it must not be mixed with the variance suite.
* Shared suite has exactly 20 attacks x 3 levels = 60. Each protocol has
  12 hosts x 2 watermarks and seed 2026. All methods use the same attack
  configurations (JPEG2000 strict/no fallback; noise seed 123).
* `revision_20x3_variance.json` and `revision_20x3_sigma.json` now include
  EXACTLY the 11 methods in the supplied Table 10 (not the six of the
  previous patch). Protocol IDs bumped to `v2_table10` so old partial trial
  JSON is not mistaken for a current run.
* Nha 2022 QR uses native **4x4 RGB-blue** blocks and published R(1,1)
  embedding/extraction, q=10. This is NOT the Hessenberg baseline.
* Ariatmanto–Ernawan 2019 uses the paper's 8x8 DCT five mid-band
  coordinates and Arnold scrambling but a **disclosed 64x64 adaptation**:
  4096 bits need 820 selected blocks with five coefficient slots each;
  selected-block positions in the decoder key. DO NOT claim exact original
  paper implementation nor strictly blind detection. Record this in the table
  as `key_assisted_blind` and name as `DCT 2019 (4096-bit adapted)`.
* The proposed method itself also requires stored per-block side information;
  `key_assisted_blind` in code means original-host-free WITH a key, not
  host-free AND auxiliary-data-free. Detector classes should not be conflated.

## Install

```bash
python -m pip install -e .
python -m pytest -q
```

## 1. Full 11-method Table 10 comparison and revised Table 6 / Fig. 5

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/revision_20x3_variance.json

RUN=results/revision_20x3/dctqr_revision20x3_variance_2026_11_v2_table10
python scripts/audit_benchmark.py --run-dir "$RUN" --strict
python scripts/export_revision_fig_tables.py --run-dir "$RUN" --output results/final_tables
python scripts/export_revision_extended_experiments.py --run-dir "$RUN" --output results/final_analysis
python scripts/export_revision_extraction_gallery.py --run-dir "$RUN" \
  --output results/fig5_extracted.pdf --host tiffany.bmp --watermark Cc.logo.circle.png
```

Outputs: `table10_equal_payload_comparison.csv`, `table10_clean_control_diagnostic.csv`,
`table6_by_attack_severity.csv`, `fig5_moderate_comparison.pdf/png`, plus
per-host, per-watermark, per-severity, Gaussian variance, paired HOST-cluster bootstrap
95% CI, and detailed figures (all from the SAME validated run).

**No placeholder zeros**: exporters deliberately reject partial or failed
benchmarks. The full run is CPU-intensive, with 11 x 12 x 2 = 264 trials and
264 x 60 = 15,840 attacked extractions. Use resume or max-trials to checkpoint
but finish the complete grid before putting any result in the paper.

## 2. Fig. 4 TIFFANY + C-logo, 12 preselected moderate attacks

This independent one-pair experiment DOES NOT require all 11 methods to finish,
but it uses the EXACT same variance-based attack suite, method config and seeds.

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/tiffany_fig4_variance.json
FIG=results/revision_fig4/dctqr_fig4_tiffany_c_revision20x3_variance_2026_11_v1
python scripts/audit_benchmark.py --run-dir "$FIG" --strict
python scripts/export_revision_figure4_tiffany.py --run-dir "$FIG" \
  --output results/fig4_tiffany_c_moderate
```

The figure has 12 predetermined attack types, not 12 cherry-picked high NC
conditions. Each displayed NC/BER is numerically checked against its own saved
benchmark trial. This is a single TIFFANY + C pair, NOT Table 6 means.

## 3. Six one-factor ablations on ALL 24 pairs, same 60-condition suite

```bash
python scripts/run_revision_ablation_grid.py \
  --protocol configs/benchmark/revision_20x3_variance.json \
  --output results/final_ablation
```

Ablation code is existing `run_theory_ablation.py` (only chosen configuration
flags change, NEVER source method). Full grid emits 7 variant rows, one each
for full + six ablations, backed by 24 x 60 successful attacks each.
`--limit-pairs 1` runs only a diagnostic and is flagged incomplete.

## Scientific constraints

* Revised Table 10, Table 6 and Fig. 5 statistics must be regenerated;
  the old PSNR, mean NC, crop and rotation statements do not refer to the new
  attack experiment. PSNR may stay unchanged if embedding is unchanged, but
  comparisons must still be recomputed from one run.
* The old Fig. 5 referenced a separate comparison and **must not** be
  relabeled as primary-protocol evidence.
* For crop, **no spatial resampling** is performed. The code retains the
  original 512x512 coordinates while blacking the discarded pixels. This is
  not the same as exporting a smaller 486x486 crop; state this explicitly.
* The full 20x3 is one protocol; Fig. 4 simply displays moderate cases.
* Nha's journal Fig. 1/2 use RGB **blue channel**, 4x4; do not use legacy
  2x2 results. QR 2022 and Hessenberg 2023 remain separate rows.
* The two new baselines have been checked for clean recovery, but a full
  Table 10 attack sweep has NOT been run here.
