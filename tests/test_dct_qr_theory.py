from __future__ import annotations

import numpy as np

from qr64_certified.proposals import dct_qr_theory as theory


def _host(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)


def test_eta_is_nonzero_norm_preserving_root() -> None:
    l = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    o = np.array([1.0, -0.5, -0.5], dtype=np.float64)
    eta = theory.derived_eta()
    assert eta > 0.0
    assert np.isclose(np.linalg.norm(l + eta * o), np.linalg.norm(l), rtol=0, atol=1e-14)


def test_neumann_lift_guarantees_host_analysis_nonsingularity() -> None:
    host = _host(2)
    eta = theory.derived_eta()
    a0 = theory._unlifted_analysis_matrices(host, eta)
    lift = theory.neumann_safe_lift(a0)
    assert np.all(np.linalg.norm(a0, axis=(1, 2)) < lift)
    cert = theory.compute_theory_certificate(host, eta=eta, lift=lift)
    assert cert.all_nonsingular


def test_beta_is_below_smallest_singular_value() -> None:
    host = _host(3)
    eta = theory.derived_eta()
    a0 = theory._unlifted_analysis_matrices(host, eta)
    lift = theory.neumann_safe_lift(a0)
    a = a0 + lift * np.eye(4, dtype=np.float64)[None, :, :]
    cert = theory.compute_theory_certificate(host, eta=eta, lift=lift)
    sigma_min = np.linalg.svd(a, compute_uv=False)[:, -1]
    assert np.all(cert.beta <= sigma_min * (1.0 + 1e-12))


def test_global_coset_cannot_increase_projection_energy() -> None:
    rng = np.random.default_rng(4)
    carrier = rng.normal(size=4096)
    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    steps = rng.uniform(2.0, 12.0, size=4096)
    _encoded, _flip, stats = theory._global_coset_optimize(carrier, bits, steps)
    assert stats["optimized_projection_energy"] <= stats["original_projection_energy"] + 1e-12


def test_clean_roundtrip_is_exact() -> None:
    rng = np.random.default_rng(5)
    host = _host(5)
    watermark = (rng.integers(0, 2, size=(64, 64), dtype=np.uint8) * 255).astype(np.uint8)
    watermarked, key = theory.embed(host, watermark)
    recovered = theory.extract(watermarked, key)
    assert np.array_equal(recovered > 127, watermark > 127)
