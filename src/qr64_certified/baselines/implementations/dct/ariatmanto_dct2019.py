"""Ariatmanto & Ernawan, Multimedia Tools Appl. (2019/2020), DCT baseline.

DOI: 10.1007/s11042-019-08338-x. Embedding Section 3.1, Eqs.(4),(6)
and Rule 2; extraction Section 3.2 Eq.(8) and Rule 5.

Paper-native: select lowest-variance 8x8 image blocks; compute DCT; use the
five mid-frequency positions (4,2),(3,3),(2,4),(1,5),(2,5), with 0-based
indices as printed in its Fig.2; adapt strength from five-coefficient mean;
use Rx and Ry paper statistics; inverse Arnold.

COMMON 4096-BIT ADAPTATION (NOT EXACT PAPER REPRODUCTION): original paper
uses a 32x32 watermark and has ambiguous increment semantics for 'u' in its
five-coefficient pseudocode. We deterministically assign one bit to each of
the five coefficients, requiring ceil(4096/5)=820 variance-selected blocks.
The original paper provides no complete RGB channel embedding rule; here DCT
is applied to BT.601 luma, and luminance differences are added equally to R,G,B.
Image-dependent selected coordinates must be stored for extraction; this is
KEY-ASSISTED, not host-free strictly blind. Mode='literal' preserves the paper's printed conditional updates and can
fail to recover bits even without attacks; mode='adapt' explicitly applies a
minimum-magnitude correction for the paper's C(i)>Ry(i) detector. It is NOT a
verbatim paper reproduction; label this adapted mode in every comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.fft import dctn, idctn
from qr64_certified.baselines.implementations.transform.hess_nha2023 import (
    _arnold_period, _scramble, _descramble
)

# Coordinates stated in paper's Sec.3, indexed from 0 as in Fig.2.
FIVE = np.array([(4,2),(3,3),(2,4),(1,5),(2,5)], dtype=np.int64)
LUMA_RGB = np.array([0.299,0.587,0.114], dtype=np.float64)


@dataclass
class AriatmantoDCT2019Key:
    watermark_shape: tuple[int,int]
    selected_block_ids: np.ndarray
    arnold_iter: int
    arnold_period: int
    image_shape: tuple[int,int]
    payload_size: int
    # Selected variance positions are image-derived side information.
    channel_adapter: str = 'bt601_luma_equal_rgb_delta'
    bit_mapping_adapter: str = 'five_bits_per_selected_block'
    mode: str = 'adapt'
    detection_margin: float = 3.0


def _blocks(image: np.ndarray) -> np.ndarray:
    h,w = image.shape
    if h % 8 or w % 8:
        raise ValueError('DCT 2019 expects image dimensions divisible by 8')
    return image.reshape(h//8,8,w//8,8).transpose(0,2,1,3).reshape(-1,8,8)


def _unblocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    return blocks.reshape(h//8,w//8,8,8).transpose(0,2,1,3).reshape(h,w)


def _paper_alpha(c: np.ndarray) -> np.ndarray:
    """Paper Eq.(4): alpha = mean(C) + mean(C)/64."""
    avg = c.mean(axis=-1)
    return avg + avg/64.0


def _paper_rx(c: np.ndarray) -> np.ndarray:
    """Paper Eq.(6), indexed Rx(i) in Rule 2: |C(i)| * sum(C)/64.

    Printed sum notation conflicts with the indexed pseudocode; this
    coefficient-wise interpretation is explicitly an implementation choice.
    """
    return np.abs(c) * (c.sum(axis=-1, keepdims=True)/64.0)


def _paper_ry(c: np.ndarray) -> np.ndarray:
    """Paper Eq.(8), indexed Ry(i) in extraction: C(i) * sum(C)/64.

    The printed scalar-sum and Ry(i) pseudocode disagree. We use the indexed
    expression, and explicitly report that interpretative choice.
    """
    return c * (c.sum(axis=-1, keepdims=True)/64.0)


class AriatmantoDCT2019:
    name = 'AriatmantoErnawan2019_DCT_64x64_Adapted'

    def __init__(self, *, mode: str = 'adapt', arnold_iter: int = 10,
                 wm_size: int = 64, block_size: int = 8,
                 detection_margin: float = 3.0):
        self.mode = str(mode)
        self.detection_margin = float(detection_margin)
        if self.mode not in {"adapt", "literal"}:
            raise ValueError("DCT 2019 mode must be adapted or literal")
        if self.detection_margin <= 0 or not np.isfinite(self.detection_margin):
            raise ValueError("detection_margin must be positive")
        self.arnold_iter = int(arnold_iter)
        self.wm_size = int(wm_size)
        if block_size != 8 or wm_size != 64:
            raise ValueError('DCT 2019 adapter uses 8x8 DCT blocks and a 64x64 watermark')
        self.period = _arnold_period(wm_size)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        host = np.asarray(host_rgb, dtype=np.uint8)
        watermark = np.asarray(watermark_binary, dtype=np.uint8)
        if host.ndim != 3 or host.shape[2] != 3 or watermark.shape != (64,64):
            raise ValueError('DCT 2019 expects RGB host and 64x64 binary watermark')
        h,w = host.shape[:2]
        scalar = np.tensordot(host.astype(np.float64), LUMA_RGB, axes=([-1],[0]))
        spatial = _blocks(scalar)
        wm01 = (watermark > 127).astype(np.uint8)
        payload = _scramble(wm01, self.arnold_iter, self.period).reshape(-1)
        blocks_needed = math.ceil(payload.size/5)
        if blocks_needed > len(spatial):
            raise ValueError(f'Only {len(spatial)*5} capacity bits available, need {payload.size}')
        selected = np.argsort(spatial.var(axis=(1,2)), kind='stable')[:blocks_needed]
        picked = spatial[selected].copy()
        coef = dctn(picked, type=2, norm='ortho', axes=(-2,-1))
        carrier = coef[:,FIVE[:,0],FIVE[:,1]].copy()
        alpha = _paper_alpha(carrier)[:,None]
        rx = _paper_rx(carrier)
        bits = np.zeros((blocks_needed,5), dtype=np.uint8)
        valid = np.zeros((blocks_needed,5), dtype=bool)
        bits.flat[:payload.size] = payload
        valid.flat[:payload.size] = True
        if self.mode == "literal":
            # Literal paper conditional Rule 2, with per-coefficient Rx(i).
            # It does not guarantee that the extracted bit is correct.
            update0 = valid & (bits == 0) & ((rx - carrier) > alpha)
            update1 = valid & (bits == 1) & ((carrier - rx) > alpha)
            updated = np.where(update0, rx-alpha,
                               np.where(update1, rx+alpha, carrier))
        else:
            # DECLARED COMMON-PAYLOAD ADAPTATION (not in the 2019 paper):
            # enforce a margin for the paper's extraction test C(i)>Ry(i).
            # At selected low-variance blocks sum(C) is typically <64, hence
            # the paper's Ry(i)=C(i)*sum(C)/64 test is the sign of C(i).
            # We preserve the five-carrier DCT, Arnold permutation, variance
            # selection, and adaptive magnitude from paper Eqs.(4),(6).
            magnitude = np.maximum(self.detection_margin, np.abs(alpha) + np.abs(rx))
            updated = np.where(valid & (bits == 1), np.maximum(carrier, magnitude),
                               np.where(valid & (bits == 0), np.minimum(carrier, -magnitude), carrier))
        coef[:,FIVE[:,0],FIVE[:,1]] = updated
        reconstructed = idctn(coef, type=2, norm='ortho', axes=(-2,-1))
        # Avoid manipulating unchanged host blocks. These coefficients are
        # orthonormal DCT, so inverse gives the intended spatial delta.
        difference = np.zeros_like(spatial)
        difference[selected] = reconstructed - picked
        delta = _unblocks(difference, h, w)
        out = np.clip(np.floor(host.astype(np.float64) + delta[:,:,None] + 0.5),0,255).astype(np.uint8)
        key = AriatmantoDCT2019Key((64,64), selected.astype(np.int32),
                                   self.arnold_iter, self.period, (h,w),payload.size,
                                   mode=self.mode, detection_margin=self.detection_margin)
        return out, key

    def extract(self, image_rgb: np.ndarray, key: AriatmantoDCT2019Key, *, host_rgb=None):
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != key.image_shape:
            raise ValueError('DCT 2019 extraction requires the original image dimensions')
        luma = np.tensordot(image.astype(np.float64), LUMA_RGB, axes=([-1],[0]))
        spatial = _blocks(luma)[key.selected_block_ids]
        coef = dctn(spatial, type=2, norm='ortho', axes=(-2,-1))
        c = coef[:,FIVE[:,0],FIVE[:,1]]
        ry = _paper_ry(c)
        decoded = (c > ry).astype(np.uint8).ravel()[:key.payload_size]
        wm = _descramble(decoded.reshape(key.watermark_shape),key.arnold_iter,key.arnold_period)
        return (wm*255).astype(np.uint8)


__all__ = ['AriatmantoDCT2019','AriatmantoDCT2019Key', 'FIVE', '_paper_rx','_paper_ry']
