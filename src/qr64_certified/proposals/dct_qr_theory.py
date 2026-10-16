from __future__ import annotations

"""Theory-grounded DCT-QR proposal.

This proposal deliberately leaves the established/partially-supported DCT-QR
architecture intact and replaces only the empirically fixed decisions from the
public ``dct_qr`` path:

* eta=0.07 -> Gram-Schmidt-derived opponent coefficient;
* lift=1 -> data-derived Neumann-safe diagonal lift;
* determinant epsilon gate -> nonsingularity guaranteed by the lift;
* weighted reliability fusion / percentile scaling -> separate descriptors;
* 20/40/40 step classes -> continuous beta-normalized step allocation;
* fixed 18.25/13.25/10.25 levels -> one geometry-derived reference step;
* rho=0.45 Delta -> center-QIM (continuous margin Delta/2);
* reliability-sorted pairs / pair size 2 -> one exact global binary coset;
* gain exponent 0.90 and gain clipping -> exact gamma=1 ratio correction;
* fixed MAP lambda/iterations/diagonal weight -> degree-normalized 4-neighbour
  sequential ICM run to a fixed point.

The original ``dct_qr`` implementation is not modified.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn

try:  # pragma: no cover - numba availability is environment-specific
    from numba import prange
except Exception:  # pragma: no cover
    prange = range

from qr64_certified.common.core import arnold_transform, njit
from qr64_certified.common.types import WatermarkKey
from qr64_certified._jilp_core import engine as _eng
from qr64_certified._jilp_core.method import (
    METHOD_ID as JILP_METHOD_ID,
    PAY10_BA,
    PAY10_BB,
    _best_candidate,
    _build_key_params,
    _call_with_realtime_thread_budget,
    _make_content_adaptive_schedules,
    _normalize_jilp_key,
    _pilot_score_fast,
    _schedules_from_key,
)

from .certificate import opponent_field, qr_gain_scale
from .qr_carrier import _dct_qim_carrier, _inverse_arnold_array


METHOD_ID = "dct_qr_theory"
BLOCK_SIZE = 8


@dataclass(frozen=True)
class DCTQRTheoryConfig:
    """Configuration with no tuned constants from the red-item list.

    ``reference_step`` is optional.  If omitted, the reference step is derived
    from the 8x8 orthonormal carrier geometry as N/sqrt(2).  This corresponds
    to one field-intensity unit of worst-case RMS perturbation when the carrier
    displacement is one reference step.
    """

    seed: int = 2026
    reference_step: float | None = None
    pilot_count: int = 128
    pilot_step: float = 6.0
    pilot_rho_frac: float = 0.49
    watermark_size: int = 64

    # Existing pilot-search controls are preserved from dct_qr.  They were not
    # among the user-requested red-item removals.
    sync_quick_accept: float = 0.72
    translation_radius: int = 8
    translation_subset: int = 96

    # Scientific ablation switches. Defaults preserve the full theory method.
    use_opponent_term: bool = True
    use_adaptive_beta_steps: bool = True
    use_global_coset: bool = True
    use_gain_normalization: bool = True
    use_spatial_icm: bool = True
    use_sync_search: bool = True

    def validated(self) -> "DCTQRTheoryConfig":
        if self.watermark_size != 64:
            raise ValueError("The protocol keeps the input watermark at 64x64.")
        if self.reference_step is not None and float(self.reference_step) <= 0:
            raise ValueError("reference_step must be positive when supplied.")
        if self.pilot_count < 0:
            raise ValueError("pilot_count cannot be negative.")
        if self.pilot_step <= 0:
            raise ValueError("pilot_step must be positive.")
        if not 0 <= self.pilot_rho_frac < 0.5:
            raise ValueError("pilot_rho_frac must be in [0,0.5).")
        return self

    @classmethod
    def from_mapping(
        cls, value: "DCTQRTheoryConfig | Mapping[str, Any] | None"
    ) -> "DCTQRTheoryConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        raw = {k: v for k, v in dict(value).items() if k in allowed}
        return cls(**raw).validated()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DCTQRTheoryKey:
    base_key: WatermarkKey
    config: dict[str, Any]
    certificate_summary: dict[str, Any]

    @property
    def fully_blind(self) -> bool:
        # Host-blind/key-assisted: no original host is required at extraction.
        return True


@dataclass(frozen=True)
class TheoryCertificate:
    determinant: np.ndarray
    beta: np.ndarray
    balance: np.ndarray
    coupling: np.ndarray
    lift: float

    @property
    def all_nonsingular(self) -> bool:
        return bool(np.all(self.determinant > 0.0))

    def summary(self) -> dict[str, Any]:
        return {
            "certificate_mode": "qr_theory",
            "block_count": int(self.beta.size),
            "all_det_nonzero": self.all_nonsingular,
            "min_abs_det": float(np.min(self.determinant)),
            "median_beta": float(np.median(self.beta)),
            "median_balance": float(np.median(self.balance)),
            "median_coupling": float(np.median(self.coupling)),
            "lift": float(self.lift),
        }


def derived_eta() -> float:
    """Return the nonzero norm-preserving opponent coefficient.

    Let L=(.299,.587,.114) and O=(1,-1/2,-1/2).  Requiring the augmented
    field gradient g=L+eta O to preserve the Euclidean sensitivity of the
    BT.601 luminance gradient, ||g||_2=||L||_2, gives

        eta(2<L,O> + eta||O||_2^2)=0.

    Besides the trivial eta=0 root, the unique nonzero solution is
    eta*=-2<L,O>/||O||_2^2.  Thus the opponent term is retained without a
    tuned coefficient and without changing the minimum-norm RGB-update scale.
    """

    luminance = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    opponent = np.array([1.0, -0.5, -0.5], dtype=np.float64)
    return float(-2.0 * np.dot(luminance, opponent) / np.dot(opponent, opponent))


def reference_qim_step(block_size: int = BLOCK_SIZE) -> float:
    """Geometry-derived reference step for the differential orthonormal carrier.

    The carrier update uses two orthonormal DCT atoms.  A carrier displacement
    ``delta`` therefore has field perturbation norm sqrt(2)|delta|, hence RMS
    ``sqrt(2)|delta|/N`` on an NxN block.  Choosing Delta0=N/sqrt(2) makes a
    one-step displacement correspond to one field-intensity unit of RMS.
    """

    return float(block_size / np.sqrt(2.0))


def _unlifted_analysis_matrices(rgb: np.ndarray, eta: float) -> np.ndarray:
    field = opponent_field(np.asarray(rgb, dtype=np.uint8), eta=float(eta))
    h, w = field.shape
    if h % BLOCK_SIZE or w % BLOCK_SIZE:
        raise ValueError("Image dimensions must be divisible by 8.")
    blocks = (
        field.reshape(h // BLOCK_SIZE, BLOCK_SIZE, w // BLOCK_SIZE, BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    )
    coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    a = coeff[:, :4, :4].copy()
    a[:, 0, 0] = coeff[:, 0, 1] - coeff[:, 1, 0]
    return a


def neumann_safe_lift(unlifted: np.ndarray) -> float:
    """Choose lambda > max_b ||A_b||_2 using the computable Frobenius bound.

    Since ||A||_2 <= ||A||_F, choosing lambda strictly larger than every
    Frobenius norm guarantees ||A/lambda||_2 < 1.  Then
    A+lambda I = lambda(I+A/lambda) is nonsingular by the Neumann lemma.
    """

    a = np.asarray(unlifted, dtype=np.float64)
    max_frob = float(np.max(np.linalg.norm(a, axis=(1, 2))))
    return float(np.nextafter(max_frob, np.inf))


def compute_theory_certificate(
    rgb: np.ndarray, *, eta: float | None = None, lift: float | None = None
) -> TheoryCertificate:
    eta_value = derived_eta() if eta is None else float(eta)
    base = _unlifted_analysis_matrices(rgb, eta_value)
    lift_value = neumann_safe_lift(base) if lift is None else float(lift)
    a = base + lift_value * np.eye(4, dtype=np.float64)[None, :, :]

    det = np.abs(np.linalg.det(a))
    frob = np.linalg.norm(a, axis=(1, 2))
    beta = det / np.power(frob, 3)

    q, r = np.linalg.qr(a)
    diag_raw = np.diagonal(r, axis1=-2, axis2=-1)
    sign = np.where(diag_raw < 0.0, -1.0, 1.0)
    q = q * sign[..., None, :]
    r = sign[..., :, None] * r
    del q

    diag = np.abs(np.diagonal(r, axis1=-2, axis2=-1))
    balance = diag.min(axis=1) / diag.max(axis=1)
    upper = np.triu(r, k=1)
    coupling = np.linalg.norm(upper, axis=(1, 2)) / np.linalg.norm(r, axis=(1, 2))
    return TheoryCertificate(det, beta, balance, coupling, lift_value)


def adaptive_steps_from_beta(beta: np.ndarray, reference_step: float) -> np.ndarray:
    """Continuous weak->strong step allocation without ranks or fitted weights.

    beta is a conservative nonsingularity indicator.  The geometric mean G is
    the unique multiplicative center, and

        Delta_b = Delta_0 * G / beta_b

    gives larger spacing to smaller-beta blocks without thresholds, classes,
    percentile normalization, or learned fusion weights.
    """

    values = np.asarray(beta, dtype=np.float64).reshape(-1)
    if np.any(values <= 0.0):
        raise ValueError("beta must be strictly positive for theory step allocation.")
    geometric_mean = float(np.exp(np.mean(np.log(values))))
    return float(reference_step) * geometric_mean / values


def _global_coset_optimize(
    carrier: np.ndarray, payload_bits: np.ndarray, steps: np.ndarray
) -> tuple[np.ndarray, int, dict[str, float | int]]:
    """Select one global binary coset label by exact projection-energy minimum."""

    values = np.asarray(carrier, dtype=np.float64).reshape(-1)
    bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1) & 1
    local_steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    if not (values.size == bits.size == local_steps.size):
        raise ValueError("carrier, payload_bits, and steps must have equal length")

    cost0 = np.empty(values.size, dtype=np.float64)
    cost1 = np.empty(values.size, dtype=np.float64)
    for i, (value, step) in enumerate(zip(values, local_steps, strict=True)):
        target0 = _eng._project_qim_center(float(value), float(step), 0)
        target1 = _eng._project_qim_center(float(value), float(step), 1)
        cost0[i] = float(target0 - value) ** 2
        cost1[i] = float(target1 - value) ** 2

    energy_s0 = float(np.where(bits == 0, cost0, cost1).sum())
    energy_s1 = float(np.where(bits == 0, cost1, cost0).sum())
    flip = int(energy_s1 < energy_s0)
    encoded = np.bitwise_xor(bits, np.uint8(flip))
    optimized = min(energy_s0, energy_s1)
    return encoded, flip, {
        "global_coset_flip": int(flip),
        "original_projection_energy": energy_s0,
        "flipped_projection_energy": energy_s1,
        "optimized_projection_energy": optimized,
        "projection_energy_ratio": float(optimized / max(energy_s0, np.finfo(float).tiny)),
    }


@njit(cache=True, fastmath=True, parallel=True)
def _embed_center_variable_steps(
    out,
    indices,
    bits,
    steps,
    ba,
    bb,
    w_crop,
    block_size,
    eta,
):
    """Center-QIM plus bounded integer closure for per-block steps.

    No empirical guard fraction is used.  The continuous projection is exactly
    to the requested coset center; the integer closure only restores parity if
    uint8 rounding/clipping moves the carrier across a decision boundary.
    """

    n = indices.shape[0]
    bs = block_size
    bc = w_crop // bs
    g0 = 0.299 + eta
    g1 = 0.587 - 0.5 * eta
    g2 = 0.114 - 0.5 * eta
    gg = g0 * g0 + g1 * g1 + g2 * g2
    p0 = g0 / gg
    p1 = g1 / gg
    p2 = g2 / gg
    corrections_per_block = np.zeros(n, dtype=np.int64)
    unresolved_per_block = np.zeros(n, dtype=np.int64)

    for k in prange(n):
        step = float(steps[k])
        idx = int(indices[k])
        r0 = (idx // bc) * bs
        c0 = (idx % bc) * bs
        ca = 0.0
        cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _eng._field_rgb(out[rr, cc, 0], out[rr, cc, 1], out[rr, cc, 2], eta)
                ca += f * ba[i, j]
                cb += f * bb[i, j]
        v = 0.5 * (ca - cb)
        target = _eng._project_qim_center(v, step, int(bits[k]))
        delta = target - v

        ca2 = 0.0
        cb2 = 0.0
        for i in range(bs):
            for j in range(bs):
                d = delta * (ba[i, j] - bb[i, j])
                rr = r0 + i
                cc = c0 + j
                x0 = out[rr, cc, 0] + d * p0
                x1 = out[rr, cc, 1] + d * p1
                x2 = out[rr, cc, 2] + d * p2
                if x0 < 0.0:
                    x0 = 0.0
                elif x0 > 255.0:
                    x0 = 255.0
                if x1 < 0.0:
                    x1 = 0.0
                elif x1 > 255.0:
                    x1 = 255.0
                if x2 < 0.0:
                    x2 = 0.0
                elif x2 > 255.0:
                    x2 = 255.0
                q0 = np.uint8(np.rint(x0))
                q1 = np.uint8(np.rint(x1))
                q2 = np.uint8(np.rint(x2))
                out[rr, cc, 0] = q0
                out[rr, cc, 1] = q1
                out[rr, cc, 2] = q2
                f2 = _eng._field_rgb(q0, q1, q2, eta)
                ca2 += f2 * ba[i, j]
                cb2 += f2 * bb[i, j]
        va = 0.5 * (ca2 - cb2)

        block_corrections = 0
        # At most one unit move per channel/sample degree of freedom before we
        # stop and report unresolved status; the bound is geometry-derived.
        max_updates = bs * bs * 3
        solved = False
        for _it in range(max_updates):
            qcur = int(np.rint(va / step))
            if (qcur & 1) == int(bits[k]):
                solved = True
                break
            desired = _eng._project_qim_center(va, step, int(bits[k]))
            need = 1.0 if desired > va else -1.0
            best_abs = 0.0
            best_i = -1
            best_j = -1
            best_c = -1
            best_dir = 0
            best_s = 0.0
            for i in range(bs):
                for j in range(bs):
                    kval = 0.5 * (ba[i, j] - bb[i, j])
                    rr = r0 + i
                    cc = c0 + j
                    s0 = kval * g0
                    s1 = kval * g1
                    s2 = kval * g2
                    dr = 1 if need * s0 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 0] < 255) or (dr < 0 and out[rr, cc, 0] > 0)) and abs(s0) > best_abs:
                        best_abs = abs(s0); best_i = i; best_j = j; best_c = 0; best_dir = dr; best_s = s0
                    dr = 1 if need * s1 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 1] < 255) or (dr < 0 and out[rr, cc, 1] > 0)) and abs(s1) > best_abs:
                        best_abs = abs(s1); best_i = i; best_j = j; best_c = 1; best_dir = dr; best_s = s1
                    dr = 1 if need * s2 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 2] < 255) or (dr < 0 and out[rr, cc, 2] > 0)) and abs(s2) > best_abs:
                        best_abs = abs(s2); best_i = i; best_j = j; best_c = 2; best_dir = dr; best_s = s2
            if best_i < 0:
                break
            rr = r0 + best_i
            cc = c0 + best_j
            out[rr, cc, best_c] = np.uint8(int(out[rr, cc, best_c]) + best_dir)
            va += best_s * best_dir
            block_corrections += 1

        if not solved:
            qcur = int(np.rint(va / step))
            if (qcur & 1) != int(bits[k]):
                unresolved_per_block[k] = 1
        corrections_per_block[k] = block_corrections

    return int(np.sum(corrections_per_block)), int(np.sum(unresolved_per_block))


def _pilot_align(attacked: np.ndarray, key_params: dict[str, Any], cfg: DCTQRTheoryConfig):
    """Choose a pilot-supported candidate only when it improves pilot agreement."""

    raw = np.asarray(attacked, dtype=np.uint8)
    raw_score, raw_valid = _pilot_score_fast(raw, key_params, 0, 0)
    candidate, shift_y, shift_x, meta = _best_candidate(
        raw,
        key_params,
        quick_accept=cfg.sync_quick_accept,
        translation_radius=cfg.translation_radius,
        translation_subset=cfg.translation_subset,
    )
    aligned = np.asarray(candidate, dtype=np.uint8)
    if shift_y or shift_x:
        aligned = np.roll(aligned, shift=(-int(shift_y), -int(shift_x)), axis=(0, 1))
    candidate_score, candidate_valid = _pilot_score_fast(aligned, key_params, 0, 0)
    use_candidate = bool(candidate_score > raw_score)
    selected = aligned if use_candidate else raw
    return selected, {
        **dict(meta),
        "raw_pilot_score": float(raw_score),
        "candidate_pilot_score": float(candidate_score),
        "pilot_valid": int(candidate_valid if use_candidate else raw_valid),
        "accepted_geometric_correction": use_candidate,
    }


def _degree_normalized_icm(data: np.ndarray) -> np.ndarray:
    """Sequential four-neighbour ICM with graph-degree-normalized coupling.

    The energy uses coupling 1/Delta_G where Delta_G=4 is the maximum degree of
    the four-neighbour image grid.  Sequential coordinate updates are repeated
    until a fixed point, so no empirical iteration count or diagonal weight is
    introduced.
    """

    field = np.asarray(data, dtype=np.float64)
    state = np.where(field >= 0.0, 1, -1).astype(np.int8)
    h, w = state.shape
    coupling = 1.0 / 4.0
    changed = True
    while changed:
        changed = False
        for i in range(h):
            for j in range(w):
                neighbour_sum = 0
                if i > 0:
                    neighbour_sum += int(state[i - 1, j])
                if i + 1 < h:
                    neighbour_sum += int(state[i + 1, j])
                if j > 0:
                    neighbour_sum += int(state[i, j - 1])
                if j + 1 < w:
                    neighbour_sum += int(state[i, j + 1])
                new_state = 1 if field[i, j] + coupling * neighbour_sum >= 0.0 else -1
                if new_state != int(state[i, j]):
                    state[i, j] = new_state
                    changed = True
    return (state > 0).astype(np.uint8) * 255


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DCTQRTheoryConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    cfg = DCTQRTheoryConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    if wm.shape != (64, 64):
        raise ValueError(f"This protocol keeps the original 64x64 watermark; got {wm.shape}.")
    if host.shape[0] % BLOCK_SIZE or host.shape[1] % BLOCK_SIZE:
        raise ValueError("Host dimensions must be divisible by 8.")

    eta = derived_eta() if cfg.use_opponent_term else 0.0
    certificate = compute_theory_certificate(host, eta=eta)
    reference_step = (
        reference_qim_step(BLOCK_SIZE)
        if cfg.reference_step is None
        else float(cfg.reference_step)
    )

    params = _build_key_params(
        tuple(int(x) for x in host.shape),
        tuple(int(x) for x in wm.shape),
        seed=cfg.seed,
        step=reference_step,
        rho_frac=0.0,
        pilot_count=cfg.pilot_count,
        pilot_step=cfg.pilot_step,
        pilot_rho_frac=cfg.pilot_rho_frac,
        eta=eta,
        pilot_schedule="stratified",
        use_integer_lattice_closure=True,
        projection_mode="center",
        enable_pilots=True,
        enable_sync_search=True,
        payload_schedule_mode="random",
        scramble_payload=True,
    )
    payload_indices, pilot_indices, pilot_bits, capacity, _hc, wc, _pool = (
        _make_content_adaptive_schedules(
            host,
            tuple(int(x) for x in wm.shape),
            seed=cfg.seed,
            pilot_count=cfg.pilot_count,
            block_size=BLOCK_SIZE,
            eta=eta,
            pilot_schedule="stratified",
            payload_schedule_mode="random",
        )
    )
    if capacity != wm.size:
        raise ValueError("dct_qr_theory currently assumes one payload bit per 8x8 block.")
    params["payload_indices"] = payload_indices.astype(int).tolist()
    params["pilot_indices"] = pilot_indices.astype(int).tolist()
    params["pilot_bits"] = pilot_bits.astype(int).tolist()
    params["schedule_storage"] = "explicit_dct_qr_theory"

    output = host.copy()
    _eng._embed_one_carrier_md(
        host,
        output,
        pilot_indices,
        pilot_bits,
        _eng.PILOT_BA,
        _eng.PILOT_BB,
        float(cfg.pilot_step),
        float(cfg.pilot_step) * float(cfg.pilot_rho_frac),
        int(wc),
        BLOCK_SIZE,
        eta,
    )

    scrambled = arnold_transform(
        (wm > 127).astype(np.uint8), int(params["arnold_iter"])
    ).ravel()
    payload_beta = certificate.beta[payload_indices]
    if cfg.use_adaptive_beta_steps:
        steps = adaptive_steps_from_beta(payload_beta, reference_step)
    else:
        steps = np.full(payload_beta.shape, float(reference_step), dtype=np.float64)

    carrier = _dct_qim_carrier(output, eta)[payload_indices]
    if cfg.use_global_coset:
        encoded, global_flip, coset_stats = _global_coset_optimize(carrier, scrambled, steps)
    else:
        encoded = scrambled.copy()
        global_flip = 0
        _, _, full_coset_stats = _global_coset_optimize(carrier, scrambled, steps)
        base_energy = float(full_coset_stats["original_projection_energy"])
        coset_stats = {
            **full_coset_stats,
            "global_coset_flip": 0,
            "optimized_projection_energy": base_energy,
            "projection_energy_ratio": 1.0,
            "ablation_global_coset_disabled": True,
        }

    execution = np.argsort(payload_indices, kind="stable")
    updates, unresolved = _call_with_realtime_thread_budget(
        _embed_center_variable_steps,
        output,
        payload_indices[execution].astype(np.int32),
        encoded[execution].astype(np.uint8),
        steps[execution].astype(np.float64),
        PAY10_BA,
        PAY10_BB,
        int(wc),
        BLOCK_SIZE,
        eta,
    )
    if int(unresolved):
        raise RuntimeError(
            f"dct_qr_theory integer-lattice embedding left {int(unresolved)} unresolved blocks."
        )

    # Exact QR gain identity uses the unlifted carrier-containing first column.
    gain_reference = qr_gain_scale(output, eta=eta, lift=0.0, mode="carrier_r11")

    params.update(
        {
            "theory_variant": True,
            "eta_rule": "eta=-2<L,O>/<O,O> from ||L+eta O||_2=||L||_2 (nonzero root)",
            "analysis_lift": float(certificate.lift),
            "analysis_lift_rule": "nextafter(max_b ||A_b^0||_F, +inf)",
            "adaptive_step_by_payload": steps.astype(float).tolist(),
            "adaptive_step_rule": (
                "Delta_b=Delta_0*GM(beta)/beta_b"
                if cfg.use_adaptive_beta_steps
                else "ablation: constant Delta_b=Delta_0"
            ),
            "reference_step": float(reference_step),
            "reference_step_rule": "N/sqrt(2) from orthonormal differential-carrier RMS geometry",
            "payload_projection": "center_qim",
            "continuous_guard_margin": "Delta_b/2",
            "global_coset_flip": int(global_flip),
            "global_coset_stats": coset_stats,
            "coset_rule": (
                "global argmin over s in {0,1}"
                if cfg.use_global_coset
                else "ablation: fixed s=0"
            ),
            "decomposition_gain_reference": gain_reference.astype(float).tolist(),
            "decomposition_gain_gamma": 1.0,
            "decomposition_gain_clip": None,
            "decomposition_gain_rule": (
                "exact r11 ratio correction under positive scalar gain"
                if cfg.use_gain_normalization
                else "ablation: no r11 gain normalization"
            ),
            "integer_lattice_unit_updates": int(updates),
            "integer_lattice_unresolved": 0,
            "reliability_fusion": "none; beta, balance, and coupling remain separate descriptors",
            "step_partition": "none",
            "pairwise_reliability_sort": False,
            "map_rule": (
                "degree-normalized four-neighbour sequential ICM to fixed point"
                if cfg.use_spatial_icm
                else "ablation: direct hard QIM decisions"
            ),
            "ablation_switches": {
                "use_opponent_term": bool(cfg.use_opponent_term),
                "use_adaptive_beta_steps": bool(cfg.use_adaptive_beta_steps),
                "use_global_coset": bool(cfg.use_global_coset),
                "use_gain_normalization": bool(cfg.use_gain_normalization),
                "use_spatial_icm": bool(cfg.use_spatial_icm),
                "use_sync_search": bool(cfg.use_sync_search),
            },
        }
    )

    base_key = WatermarkKey(
        method_id=JILP_METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=cfg.seed,
        repeat=1,
        step=float(reference_step),
        arnold_iter=int(params["arnold_iter"]),
        arnold_period=int(params["arnold_period"]),
        threshold=127,
        params=params,
        schedule=[],
        fully_blind=True,
        side_information=(
            "host-blind, key-assisted: per-payload theory steps, one global coset bit, "
            "QR gain reference, and keyed schedules; original host is not required"
        ),
    )
    key = DCTQRTheoryKey(base_key, cfg.to_dict(), certificate.summary())
    metadata = {
        "method_id": METHOD_ID,
        "eta": eta,
        "reference_step": reference_step,
        "lift": certificate.lift,
        "global_coset_flip": global_flip,
        "coset": coset_stats,
        "certificate": certificate.summary(),
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: DCTQRTheoryKey,
    *,
    return_metadata: bool = False,
):
    cfg = DCTQRTheoryConfig.from_mapping(key.config)
    params = _normalize_jilp_key(key.base_key)
    attacked = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if cfg.use_sync_search:
        aligned, sync_meta = _pilot_align(attacked, params, cfg)
    else:
        aligned = attacked
        sync_meta = {"accepted_geometric_correction": False, "ablation_sync_search_disabled": True}

    payload_indices, *_ = _schedules_from_key(params)
    steps = np.asarray(params["adaptive_step_by_payload"], dtype=np.float64)
    eta = float(params["eta"])
    reference = np.asarray(params["decomposition_gain_reference"], dtype=np.float64)
    current = qr_gain_scale(aligned, eta=eta, lift=0.0, mode="carrier_r11")
    tiny = np.finfo(np.float64).tiny
    ratio = current / np.maximum(reference, tiny)
    ratio = np.maximum(ratio, tiny)

    carrier = _dct_qim_carrier(aligned, eta)[payload_indices]
    if cfg.use_gain_normalization:
        corrected = carrier / ratio[payload_indices]  # gamma = 1 exactly
    else:
        corrected = carrier
    units = corrected / steps
    lattice = np.rint(units)
    scrambled_bits = (lattice.astype(np.int64) & 1).astype(np.uint8)
    qim_confidence = np.clip(1.0 - 2.0 * np.abs(units - lattice), 0.0, 1.0)

    global_flip = np.uint8(int(params.get("global_coset_flip", 0)) & 1)
    scrambled_bits = np.bitwise_xor(scrambled_bits, global_flip)

    wm_shape = tuple(int(x) for x in params["watermark_shape"])
    bit_map = _inverse_arnold_array(scrambled_bits.reshape(wm_shape), params)
    conf_map = _inverse_arnold_array(qim_confidence.reshape(wm_shape), params)

    # The data term is already dimensionless in [-1,1].  Keep the established
    # Ising/ICM prior but remove fitted lambda, diagonal weight, and iteration cap.
    sign = np.where(bit_map > 0, 1.0, -1.0)
    data = sign * conf_map

    # On exact/no-attack decoding, center-QIM plus integer closure should be used
    # directly; the spatial prior is only an uncertainty regularizer.
    identity_error = float(np.max(np.abs(np.log(ratio[payload_indices]))))
    machine_identity = float(np.sqrt(np.finfo(np.float64).eps))
    if identity_error <= machine_identity or not cfg.use_spatial_icm:
        recovered = (bit_map > 0).astype(np.uint8) * 255
        inference_path = (
            "exact_gain_identity" if identity_error <= machine_identity else "hard_qim_ablation"
        )
    else:
        recovered = _degree_normalized_icm(data)
        inference_path = "degree_normalized_icm"

    certificate = compute_theory_certificate(
        aligned,
        eta=eta,
        lift=float(params["analysis_lift"]),
    )
    metadata = {
        "method_id": METHOD_ID,
        "original_host_used": False,
        "original_watermark_used": False,
        "extraction_model": "host-blind, key-assisted",
        "sync": sync_meta,
        "mean_qim_confidence": float(np.mean(conf_map)),
        "gain_ratio_mean": float(np.mean(ratio[payload_indices])),
        "gain_ratio_std": float(np.std(ratio[payload_indices])),
        "gain_identity_error": identity_error,
        "gain_gamma": 1.0,
        "gain_clipped": False,
        "inference_path": inference_path,
        "certificate": certificate.summary(),
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "METHOD_ID",
    "DCTQRTheoryConfig",
    "DCTQRTheoryKey",
    "TheoryCertificate",
    "derived_eta",
    "reference_qim_step",
    "neumann_safe_lift",
    "compute_theory_certificate",
    "adaptive_steps_from_beta",
    "embed",
    "extract",
]
