"""Only the two shared mathematical helpers used by corrected DCT-QR theory.

Extracted unchanged from the legacy DCT-QR implementation; the original
DCT-QR proposal is intentionally NOT shipped in this package.
"""
import numpy as np
from scipy.fft import dctn
from .certificate import opponent_field

def _arnold_permute_preserve_dtype(array: np.ndarray, iterations: int) -> np.ndarray:
    src = np.asarray(array).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError("Arnold transform requires a square array.")
    n = src.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        dst = np.empty_like(src)
        dst[rr, cc] = src
        src = dst
    return src

def _inverse_arnold_array(array: np.ndarray, key_params: dict) -> np.ndarray:
    if not bool(key_params.get("scramble_payload", True)):
        return np.asarray(array)
    period = int(key_params["arnold_period"])
    iterations = int(key_params["arnold_iter"])
    inverse_iterations = (period - (iterations % period)) % period
    return _arnold_permute_preserve_dtype(np.asarray(array), inverse_iterations)

def _dct_qim_carrier(image: np.ndarray, eta: float) -> np.ndarray:
    """Differential 8x8 orthonormal DCT carrier of the corrected theory."""
    field = opponent_field(np.asarray(image, dtype=np.uint8), eta=float(eta))
    h, w = field.shape
    blocks = (field.reshape(h // 8, 8, w // 8, 8)
              .transpose(0, 2, 1, 3).reshape(-1, 8, 8))
    coefficients = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    return 0.5 * (coefficients[:, 0, 1] - coefficients[:, 1, 0])
