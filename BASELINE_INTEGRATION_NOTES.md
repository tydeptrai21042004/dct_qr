# Adding Nha QR 2022 and Ariatmanto–Ernawan DCT 2020

These are **not** implemented or benchmarked in this patch. Do not enter rows of
zero-valued PSNR/NC/BER to represent them and do not copy the results of an old
2x2 QR run into a new 4x4 method.

## QR Nha is NOT the existing Hessenberg Nha baseline

The registered `hessenberg_nha2023_blind` implementation at
`src/qr64_certified/baselines/implementations/transform/hess_nha2023.py`
uses a 4x4 **Hessenberg factorization**, quantizing H(2,2). It is not the QR
method in Phuong Thi Nha, Ta Minh Thanh and Nguyen Tuan Phong,
*Consideration of a robust watermarking algorithm for color image using
improved QR decomposition*, Soft Computing 26 (2022), 5069–5093,
DOI 10.1007/s00500-022-06975-3.

## Implementation checklist for each independent paper baseline

1. Add `src/qr64_certified/baselines/implementations/transform/qr_nha2022.py`
   (and a different `.../dct_ariatmanto_ernawan2020.py`). Implement `embed`
   and `extract` exactly from the selected paper, documenting all assumptions.
2. For Nha QR specify the intended block size in the code and saved key
   (`block_size=4` if the authorized adapted method is 4x4); assert that size
   in every embed/extract and in a regression test. Never reuse QR 2x2 data.
3. Add a constructor branch in `src/qr64_certified/baselines/registry.py`,
   metadata in `src/qr64_certified/baselines/metadata.py`, and a stable unique
   method ID. Mark native vs adapted fidelity and key-assisted/host-blindness
   carefully. Declare `common_4096_payload=True` only after verifying exact
   64x64 recovery and capacity.
4. Register explicit parameters in
   `configs/benchmark/baseline_parameters.json` so the block size/strength
   appear in each trial's configuration.
5. Unit tests must cover: 64x64 clean extraction, original image absent in
   blind extraction, bit ordering/Arnold inverse, boundary/clipping,
   attack-specific recoverability, and duplicate source method detection.
6. Run from a clean results directory with **the same** 12 hosts, two
   watermark images, seed 2026 and all 60 attack IDs. Export Table 5 and Fig. 5
   only after complete audits pass.

## Extra care for Ariatmanto–Ernawan DCT

Original paper: *An improved robust image watermarking by using different
embedding strengths*, Multimedia Tools and Applications 79 (2020), 12041–12067,
DOI 10.1007/s11042-019-08338-x. This is the paper supplied as `dct_2019.pdf`.
The original scheme uses 8x8 blocks, variance-selected locations, five selected
DCT coefficients C(4,2), C(3,3), C(2,4), C(1,5), C(2,5), an Arnold transform,
and stored block coordinates at extraction. It originally uses a 32x32 logo.
The prose in the abstract describes **highest** variance, while embedding
Step 2 describes **lowest** variance: clarify the intended selection before
claiming a faithful reproduction. The 64x64 project watermark requires an
explicit, disclosed capacity adaptation. Label the output *adapted
reimplementation* unless equivalence to published source code is validated.
Some literature describes this scheme as not strictly blind due to the stored
selected-block coordinates; at minimum, classify the side information and
its size explicitly.
