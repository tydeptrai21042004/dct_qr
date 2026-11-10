"""Nha, Thanh & Phong (Soft Computing, 2022) improved-QR watermarking.

Paper: DOI 10.1007/s00500-022-06975-3.  Section 3, Eqs. (22)--(25).
The paper's embedding alters only the first column of each blue-channel 4x4
block: A'=QR' with R' differing from R only at r11 implies
A'[:,0] = A[:,0] * r11'/r11 (for r11>0).  This uses the paper's
r11 = ||A[:,0]|| instead of executing a costly full QR on each block.

COMMON-PAYLOAD ADAPTER: 64x64 instead of the paper's 32x32 watermark.
Every bit is tiled across the 128x128 grid of 4x4 blocks (four copies per
bit for a 512x512 image), with majority-vote extraction.  The redundancy
is disclosed and must not be mistaken for exact paper reproduction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .hess_nha2023 import _arnold_period, _scramble, _descramble


@dataclass
class NhaQR2022Key:
    watermark_shape: tuple[int, int]
    block_size: int
    q: float
    arnold_iter: int
    arnold_period: int
    repeats: tuple[int, int]
    adjusted_negative_targets: int = 0
    zero_column_blocks: int = 0


def _r11_grid(blue: np.ndarray, block_size: int = 4) -> np.ndarray:
    h, w = blue.shape
    if h % block_size or w % block_size:
        raise ValueError('Blue image dimensions must be multiples of block size')
    blocks = np.asarray(blue, dtype=np.float64).reshape(h//4, 4, w//4, 4)
    return np.linalg.norm(blocks[:, :, :, 0], axis=1)


def _embedding_r11_target(r11: np.ndarray, bits: np.ndarray, q: float) -> tuple[np.ndarray, int]:
    """Direct vectorization of Nha Eqs. (22),(23), with positive-norm guard."""
    z = np.mod(r11, q)
    new0 = np.where(z < 0.75*q, r11 + q/4 - z, r11 + 1.25*q - z)
    new1 = np.where(z < 0.25*q, r11 - q/4 - z, r11 + 0.75*q - z)
    out = np.where(bits == 0, new0, new1)
    # A nonnegative first-column norm is required.  This only affects the
    # degenerate near-zero blocks and preserves the intended modulo-q class.
    neg = out <= 0
    out = np.where(neg, out + q * (np.floor(-out/q) + 1), out)
    return out, int(np.count_nonzero(neg))


class NhaQR2022:
    """4x4 B-channel QR r11 embedding (q=10) + 64x64 replication adapter."""
    name = 'Nha2022_ImprovedQR_64x64'

    def __init__(self, *, mode: str = 'adapt', q: float = 10.0,
                 arnold_iter: int = 10, block_size: int = 4, wm_size: int = 64):
        self.mode = str(mode)
        self.q = float(q)
        self.arnold_iter = int(arnold_iter)
        self.block_size = int(block_size)
        self.wm_size = int(wm_size)
        if self.block_size != 4 or self.wm_size != 64:
            raise ValueError('Nha QR baseline uses 4x4 host blocks and 64x64 project adapter')
        if not np.isfinite(self.q) or self.q <= 0:
            raise ValueError('q must be finite and positive')
        self.period = _arnold_period(self.wm_size)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        host = np.asarray(host_rgb, dtype=np.uint8)
        wm = np.asarray(watermark_binary, dtype=np.uint8)
        if host.ndim != 3 or host.shape[2] != 3 or wm.shape != (64,64):
            raise ValueError('Nha QR expects RGB image and 64x64 binary watermark')
        h, w = host.shape[:2]
        if h % 4 or w % 4 or (h//4) % 64 or (w//4) % 64:
            raise ValueError('4x4 block grid must be an integer multiple of the 64x64 payload')
        repeats = (h//4//64, w//4//64)
        scrambled = _scramble((wm>127).astype(np.uint8), self.arnold_iter, self.period)
        bits = np.tile(scrambled, repeats)
        b = np.asarray(host[:, :, 2], dtype=np.float64).copy().reshape(h//4,4,w//4,4)
        first_col = b[:,:,:,0]
        r11 = np.linalg.norm(first_col, axis=1)
        target, adjusted = _embedding_r11_target(r11, bits, self.q)
        ratio = np.divide(target, r11, out=np.zeros_like(target), where=r11>1.e-12)
        b[:,:,:,0] = first_col * ratio[:,None,:]
        zero = r11 <= 1.e-12
        b[:,0,:,0] = np.where(zero, target, b[:,0,:,0])
        out = host.copy()
        out[:,:,2] = np.clip(np.floor(b.reshape(h,w)+0.5), 0, 255).astype(np.uint8)
        key = NhaQR2022Key((64,64),4,self.q,self.arnold_iter,self.period,repeats,
                           adjusted_negative_targets=adjusted,
                           zero_column_blocks=int(np.count_nonzero(zero)))
        return out, key

    def extract(self, image_rgb: np.ndarray, key: NhaQR2022Key, *, host_rgb=None):
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Nha QR expects RGB image')
        if image.shape[0]//4 != 64*key.repeats[0] or image.shape[1]//4 != 64*key.repeats[1]:
            raise ValueError('Nha QR extraction requires unchanged spatial dimensions')
        r11 = _r11_grid(image[:,:,2], key.block_size)
        votes = (np.mod(r11, key.q) >= key.q/2).astype(np.uint8)
        nr, nc = key.repeats
        votes = votes.reshape(nr,64,nc,64)
        # A tie is assigned according to the first copy (deterministic).
        sums = votes.sum(axis=(0,2))
        majority = (sums > nr*nc/2).astype(np.uint8)
        ties = (sums == nr*nc/2)
        majority[ties] = votes[0,:,0,:][ties]
        restored = _descramble(majority, key.arnold_iter, key.arnold_period)
        return (restored * 255).astype(np.uint8)


__all__ = ['NhaQR2022', 'NhaQR2022Key', '_embedding_r11_target']
