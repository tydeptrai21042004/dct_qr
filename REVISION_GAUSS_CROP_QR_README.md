# Revision patch: Gaussian, crop, rotation, and consistent Fig. 5/Tables 5–6

This package changes **only the experimental infrastructure**. It does not
implement, modify, claim results for, or register the missing Nha QR (2022) and
Ariatmanto–Ernawan DCT (2019) baselines. Those two require separate validated
implementations of the primary-paper algorithms. The existing
`hessenberg_nha2023_blind` is **Hessenberg**, not Nha QR.

## Fixed primary experiment (60 attack conditions)

- 20 families × 3 severities = 60 cases, without clean controls.
- JPEG2000 requires a functioning codec in the revised suite; it will fail
  visibly instead of silently substituting ordinary JPEG.
- Rotation is a *single* 5°, 10°, 15° rotation, **not rotation + inverse rotation**.
- Crop is `crop_zero_pad`: retains the centered 95%, 90%, 80% of rows and columns
  on the **original-size zero-filled canvas, with no resampling**. It is NOT a
  standalone smaller image. The latter is available as `true_crop_only` for a
  future variable-size extractor and metric protocol. Caption it **center crop,
  zero-filled outside retained area**; don't label it crop-and-resize.
- TWO DIFFERENT, separately versioned Gaussian definitions are provided:
  `revision_20x3_sigma`: sigma = 1, 3, 5 in 8-bit units;
  `revision_20x3_variance`: variance = 0.001, 0.003, 0.005 in normalized [0,1] units.
  Never pool the two or present old table values for either.
- Only the 12 named hosts from the manuscript are included; `lenna.bmp` is
  explicitly excluded by `host_files`, by the global image listing, and by
  an explicit guard in `load_host_rgb`. No Lena is used even in single-image scripts. Both 64×64 watermarks and seed 2026 are
  retained. The default comparison list contains **only methods already in
  the repository**. Update the methods list only after verified new baselines
  are integrated.

## Usage (repository root)

```bash
python -m pip install -e .
python -m pytest -q tests/test_revision_20x3_protocol.py

# First smoke run: one host, one method, only 6 attacks (disposable new output location)
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/revision_20x3_variance.json \
  --methods dct_qr_theory --host-limit 1 --attack-limit 6 \
  --output-dir results/revision_smoke --max-trials 1

# Full comparison: 12 hosts x 2 watermarks x 4 methods, 60 attacks each
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/revision_20x3_variance.json

# Check completeness (strict)
python scripts/audit_benchmark.py \
  --run-dir results/revision_20x3/dctqr_revision20x3_variance_2026_11_v1 --strict

# Generate tables and Fig. 5 from the exact same validated trial files
python scripts/export_revision_fig_tables.py \
  --run-dir results/revision_20x3/dctqr_revision20x3_variance_2026_11_v1 \
  --output results/revision_20x3/revision_tables_figs
```

The runner supports `--max-trials`, checkpoints, and `--resume` through the
configuration. `export_revision_fig_tables.py` *refuses* to produce a purported
complete table when any host/method/attack is missing, failed or nonfinite.
Missing methods are never shown as rows of zero metrics.

**Scientific disclosure:** this is a new benchmark (different Gaussian levels,
rotation geometry, and crop definition) so it is not numerically comparable to
previous results without rerunning *every* included method. The figure produced
is a **moderate-attack mean-NC line plot**; it replaces the prior unrelated
visual reference with results from exactly the same method/host/watermark
combinations as Tables 5–6. For illustrative extracted-bit panels, regenerate
using the same `revision_20x3_variance` suite and label the selected host and
watermark; do not mistake a single-pair example for a 24-pair aggregate.

## Source-paper caveat

Ariatmanto and Ernawan, *An improved robust image watermarking by using
 different embedding strengths*, published online in January 2020
(DOI 10.1007/s11042-019-08338-x), uses 8×8 DCT blocks, five middle-frequency
coefficients, variance-selected locations, Arnold scrambling and stored block
coordinates. Its original study used 32×32 watermarks. A 64×64 adaptation would
need explicit capacity, location selection and keyed-extraction documentation;
it cannot be honestly reported as an identical original-paper rerun.

## Optional: extracted-watermark comparison panels for Fig. 5

Once the **full, strict** benchmark is complete, generate a visual Fig. 5 for
Pepper + C-logo and exactly the 3 *moderate* Gaussian/rotation/crop conditions:

```bash
python scripts/export_revision_extraction_gallery.py \
  --run-dir results/revision_20x3/dctqr_revision20x3_variance_2026_11_v1 \
  --output results/revision_20x3/fig5_extraction_methods
```

For every displayed NC and BER, the gallery re-extracts the watermark and
asserts **exact agreement** with the corresponding stored benchmark row. Its
single-pair values will *not* equal Table 6's 24-pair means, which is expected.

## Scope & scientific integrity

**Unchanged proposed method:** No file under `src/qr64_certified/proposals/`,
`src/qr64_certified/_jilp_core/`, or the main method configuration is edited.
This patch does not improve the method's Gaussian or rotation robustness by
changing its algorithm. The revised benchmark may reveal weaker robustness.

**Important:** The user-requested QR Nha baseline is NOT the existing
`hessenberg_nha2023_blind` implementation. The latter is a separate Hessenberg
baseline. Do not relabel it as QR. The QR Nha code mentioned in chat is not
included in the supplied repository; no unverified algorithm or fabricated
row has been added. The 2019 DCT paper is available as a PDF, but source code
and a validated equal-payload adaptation are not supplied; it is also **not**
included as a completed baseline. The added comparison exporter lists only
actually integrated and successfully tested methods, and refuses partial runs.

**Crop definition:** `crop_zero_pad` masks image borders outside the original
central rectangle without resizing. This fixed-canvas evaluation is distinct
from delivering an actual reduced-size cropped image to the extractor. The
latter is implemented as `true_crop_only` but cannot be used with the current
fixed-512x512 extractor without changing the attack evaluation contract.

**Figure 4:** The original requested Tiffany + C-logo illustration is
separate from Fig. 5 and can be generated with
`--host tiffany.bmp --watermark Cc.logo.circle.png` using the extraction
gallery script, but that gallery shows only 3 designated moderate attacks.
`fig_Tiffani.pdf` is not bundled or changed here.

## Figure 4: Tiffany with C-logo, moderate attacks only

Once the complete revision benchmark has passed the strict audit:

```bash
python scripts/export_revision_figure4_tiffany.py \
  --run-dir results/revision_20x3/dctqr_revision20x3_variance_2026_11_v1 \
  --output results/revision_20x3/fig4_tiffany_c
```

The generator displays 12 **preselected** moderate attack families, each
with its attacked Tiffany image and recovered C-logo. It writes PDF, PNG, CSV,
and provenance JSON. The NC and BER in every panel must agree with the matching
stored trial. This illustration is one image/one watermark and should not be
presented as the 12-host aggregate in Table 6. No Lena image is loaded.
