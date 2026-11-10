# Two baseline implementations for the DCT–QR comparison (November 2026)

This ZIP is a **source-code patch**, not computed Table 10 values. It can be
applied directly over the original `dct_qr-main (1).zip` repository; it includes
the earlier experimental-protocol patch so it does **not** require the earlier
ZIP to be installed first. Only added or modified source/config/test files are
included. No proposal method files are modified, no dataset/host/watermark is
included, and **Lena/Lenna is excluded by the common image loader**.

## Added methods

| Registry method ID | Paper | What is implemented / caveat |
|---|---|---|
| `qr_nha2022_blind` | Phuong Thi Nha, Ta Minh Thanh, Nguyen Tuan Phong, *Soft Computing* (2022), DOI 10.1007/s00500-022-06975-3 | 4x4 blue-channel first-column r11=||a1||, Sun-type paper Eqs.(22),(23),(25), q=10, Arnold; 64x64 4-way tiling and majority voting **adapted** from native 32x32 watermark. First-column-only update is algebraically equivalent to QR's R(1,1) update when r11>0. |
| `dct_adaptive_ariatmanto2019_keyassisted` | Dhani Ariatmanto, Ferda Ernawan, *Multimedia Tools and Applications* (2019 online/2020 issue), DOI 10.1007/s11042-019-08338-x | 8x8 DCT, 5 stated midband positions, lowest-variance blocks, Eqs.(4),(6),(8), Arnold, and its extraction inequality. **Not an exact paper reproduction:** 64x64 uses 5-bit/group mapping (820 blocks); paper's indexed Rx(i)/Ry(i) notation conflicts with its printed sums; color is BT.601 luma with equal RGB deltas. Default `mode=adapt` adds an explicitly disclosed detector-margin correction so clean payload is recoverable. `mode=literal` uses printed Rule 2 without that correction (clean BER can be large). Coordinates selected using original image variance are carried in the key; original host pixels are not needed at extraction, but it is **key-assisted** rather than strictly blind. |

**Do not label the second method as an exact Ariatmanto–Ernawan 2019
reproduction or strictly blind.** The paper does not provide an unambiguous
4096-bit embedding map or working code. This adaptation is an explicit,
reproducible image-benchmark operating point, not evidence that the original
paper reports the same performance. No PSNR, NC or BER is invented.

The two new methods are independent of the corrected proposal. We did not
change `src/qr64_certified/proposals/`, the proposed QIM/QR math or its
config. The original Nha-Hessenberg 2023 baseline remains separate.

## Installation and tests

Unzip this patch into the *root of the original repository* (the directory
which has `src/`, `configs/`, `scripts/`, `tests/`), overwriting the paths in
the patch. Run from that root:

```bash
python -m pip install -e .
python -m pytest -q
```

## Run ONLY the two new baselines, 12 hosts, W1 and W2, 20 x 3 attacks

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/two_new_baselines_20x3.json
```

This creates a separate output directory and does not combine with previous
runs. For a short smoke check before the full sweep:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/two_new_baselines_20x3.json \
  --host-limit 1 --attack-limit 3 --max-trials 4 \
  --output-dir results/two_baseline_smoke
```

## Run the proposed method PLUS comparison methods in one experiment

`configs/benchmark/revision_20x3_variance.json` (and `_sigma.json`) now
includes both new baseline IDs alongside the proposal and existing listed
baselines. For the revised variance protocol:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/revision_20x3_variance.json
```

The original 12 non-Lena hosts and the two 64x64 watermarks are fixed in
these protocol JSON files. The revised attacks use Gaussian **variance**
(`0.001/0.003/0.005`) or the alternate **sigma** (`1/3/5`) protocol, direct
rotation 5/10/15 degrees, and zero-padded crop without resizing. Keep a
single protocol for figures and tables.

After the combined run is complete, adjust `RUN` to the actual directory
`<output_dir>/<protocol_id>` and export:

```bash
RUN=results/revision_20x3/dctqr_revision20x3_variance_2026_11_v1
python scripts/audit_benchmark.py --run-dir "$RUN" --strict
python scripts/export_revision_fig_tables.py --run-dir "$RUN" \
  --output results/revision_tables
```

Outputs include `table10_equal_payload_comparison.csv`,
`table10_clean_control_diagnostic.csv`, `table5_comparison.csv`,
`table6_by_attack_severity.csv`, and `fig5_moderate_comparison.pdf`.
Those outputs are produced from the **same** validated trial rows; a missing
or failed trial aborts publication export instead of filling cells with zero.
No combined-table file is produced by the new-baselines-only protocol,
because Table 6 requires the proposed method in the same run.

## Validation performed when preparing the patch

- `pytest -q`: **27 passed**.
- 4 end-to-end sample benchmark trials (1 host x 2 watermarks x 2 new
  baselines) with 3 attacks each: **all audit PASS**.
- All 12 allowed non-Lena hosts x both original watermarks (24 clean cases)
  checked with both methods. Nha-QR: mean clean NC=1, BER=0; DCT adapted:
  mean clean NC about 0.99982248 and mean BER about 0.00022380.
- No full 12 x 2 x 60 attack sweep is claimed; fill Table 10 **only after**
  that sweep is actually complete.

## Paper caveats

Nha 2022's core embedding uses blue-channel 4x4 blocks, not the corrected
proposal's distinct QR-derived matrix. The *proposed* DCT-QR implementation
is never changed. DCT 2019's printed embedding rules are insufficient to
ensure robust bit recovery under a 64x64 one-bit-per-slot mapping. The
explicit adapter corrects the decision margin to make a meaningful,
reproducible equal-payload experiment; its result must be labeled adapted.
