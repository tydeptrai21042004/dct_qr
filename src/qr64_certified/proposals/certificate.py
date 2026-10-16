"""Mathematical field, QR, and gain primitives for corrected DCT-QR theory only.

The other Schur/proposal certificate implementations are intentionally removed.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import dctn

def opponent_field(rgb: np.ndarray, eta: float = 0.07) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {x.shape}.")
    return (
        0.299 * x[..., 0]
        + 0.587 * x[..., 1]
        + 0.114 * x[..., 2]
        + float(eta) * (x[..., 0] - 0.5 * x[..., 1] - 0.5 * x[..., 2])
    )


def analysis_matrices(rgb: np.ndarray, eta: float = 0.07, lift: float = 1.0) -> np.ndarray:
    """Build one lifted 4x4 low-frequency analysis matrix per 8x8 block.

    The DC term is replaced by an antisymmetric low-frequency difference so the
    certificate measures local directional structure rather than mean intensity.
    The diagonal lift is virtual: it is used only for certificate analysis and
    is never embedded into the image.
    """
    f = opponent_field(rgb, eta=eta)
    h, w = f.shape
    if h % 8 or w % 8:
        raise ValueError("Image dimensions must be divisible by 8.")
    blocks = f.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    a = coeff[:, :4, :4].copy()
    a[:, 0, 0] = coeff[:, 0, 1] - coeff[:, 1, 0]
    a += float(lift) * np.eye(4, dtype=np.float64)[None, :, :]
    return a


def canonical_qr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(a, dtype=np.float64))
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    sign = np.where(diag < 0.0, -1.0, 1.0)
    q = q * sign[..., None, :]
    r = sign[..., :, None] * r
    return q, r


def qr_gain_scale(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
    mode: str = "diag_l2",
) -> np.ndarray:
    """Return a positive QR scale for local attack-gain estimation.

    ``diag_l2`` reproduces the original global scale
    ``||diag(R)||_2`` on the lifted analysis matrix.

    ``carrier_r11`` uses the canonical first diagonal factor of an unlifted
    analysis matrix. Since ``r_11 = ||A[:,0]||_2``, it measures attenuation in
    the low-frequency column that contains the DCT-QIM carrier rather than
    averaging over unrelated QR directions.
    """
    normalized = str(mode).strip().lower()
    if normalized == "carrier_r11":
        matrices = analysis_matrices(rgb, eta=eta, lift=0.0)
        _q, r = canonical_qr(matrices)
        return np.abs(r[:, 0, 0]) + 1e-12
    if normalized != "diag_l2":
        raise ValueError("QR gain mode must be 'diag_l2' or 'carrier_r11'.")
    matrices = analysis_matrices(rgb, eta=eta, lift=lift)
    _q, r = canonical_qr(matrices)
    diagonal = np.abs(np.diagonal(r, axis1=-2, axis2=-1))
    return np.sqrt(np.sum(diagonal * diagonal, axis=1) + 1e-12)
