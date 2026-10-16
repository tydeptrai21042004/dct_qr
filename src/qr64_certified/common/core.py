from __future__ import annotations

from functools import lru_cache
import math
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


@lru_cache(maxsize=32)
def arnold_period(n: int, max_iter: int = 8192) -> int:
    common = {16: 12, 32: 24, 64: 48, 128: 96, 256: 192}
    if int(n) in common:
        return common[int(n)]
    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)
    cur = idx.copy()
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for p in range(1, max_iter + 1):
        dst = np.empty_like(cur)
        dst[rr, cc] = cur
        cur = dst
        if np.array_equal(cur, idx):
            return p
    raise RuntimeError(f"Arnold period not found for n={n}")


def arnold_transform(mat01: np.ndarray, iterations: int) -> np.ndarray:
    src = np.asarray(mat01, dtype=np.uint8).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError("Arnold transform requires a square watermark")
    n = src.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        dst = np.empty_like(src)
        dst[rr, cc] = src
        src = dst
    return src


def arnold_inverse(mat01: np.ndarray, iterations: int, period: int) -> np.ndarray:
    return arnold_transform(mat01, (int(period) - (int(iterations) % int(period))) % int(period))


def make_block_indices(h_crop: int, w_crop: int, block_size: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    br = h_crop // block_size
    bc = w_crop // block_size
    capacity = br * bc
    needed = int(payload_bits) * int(repeat)
    if needed > capacity:
        raise ValueError(f"Not enough blocks: need {needed}, capacity {capacity}")
    rng = np.random.default_rng(int(seed))
    return rng.permutation(capacity)[:needed].astype(np.int32)


def make_freq_indices(h: int, w: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    coords: list[tuple[int, int]] = []
    u0, u1 = max(4, h // 32), max(8, h // 4)
    v0, v1 = max(4, w // 32), max(8, w // 4)
    for u in range(u0, u1):
        for v in range(v0, v1):
            ru = u / h
            rv = v / w
            r = math.sqrt(ru * ru + rv * rv)
            if 0.03 <= r <= 0.20:
                coords.append((u, v))
    needed = payload_bits * repeat
    if len(coords) < needed:
        raise ValueError(f"Not enough DFT coordinates: need {needed}, capacity {len(coords)}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(coords))[:needed]
    return np.asarray([coords[i] for i in order], dtype=np.int32)


@lru_cache(maxsize=128)
def dct_basis(block_size: int, u: int, v: int) -> np.ndarray:
    bs = int(block_size)
    x = np.arange(bs, dtype=np.float32)
    y = np.arange(bs, dtype=np.float32)
    au = np.sqrt(1.0 / bs) if u == 0 else np.sqrt(2.0 / bs)
    av = np.sqrt(1.0 / bs) if v == 0 else np.sqrt(2.0 / bs)
    bx = au * np.cos(np.pi * (2 * x + 1) * u / (2 * bs))
    by = av * np.cos(np.pi * (2 * y + 1) * v / (2 * bs))
    return np.outer(bx, by).astype(np.float32)
