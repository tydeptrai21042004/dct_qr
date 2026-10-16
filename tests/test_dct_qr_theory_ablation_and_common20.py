from __future__ import annotations

from dataclasses import replace

from qr64_certified import DCT_QR_THEORY
from qr64_certified.attacks.presets import COMMON_20, COMMON_20_GROUPS, get_attack_suite
from qr64_certified.proposals.dct_qr_theory import DCTQRTheoryConfig
from qr64_certified.proposals.flags import ABLATION_FLAGS, apply_proposal_flags


def test_common20_is_exactly_twenty_and_balanced_by_group() -> None:
    assert len(COMMON_20) == 20
    assert get_attack_suite("common20") == COMMON_20
    assert {name: len(items) for name, items in COMMON_20_GROUPS.items()} == {
        "compression_quantization": 4,
        "noise": 4,
        "filtering_enhancement": 4,
        "geometric_resampling": 4,
        "photometric_point_processing": 4,
    }
    assert len({a.attack_id for a in COMMON_20}) == 20


def test_theory_ablation_flags_toggle_one_component() -> None:
    base = DCTQRTheoryConfig()
    expected = {
        "no_opponent_term": "use_opponent_term",
        "uniform_step": "use_adaptive_beta_steps",
        "no_global_coset": "use_global_coset",
        "no_gain_normalization": "use_gain_normalization",
        "no_spatial_icm": "use_spatial_icm",
        "no_sync_search": "use_sync_search",
    }
    assert set(ABLATION_FLAGS[DCT_QR_THEORY]) == set(expected)
    for ablation, field in expected.items():
        cfg, report = apply_proposal_flags(DCT_QR_THEORY, base, {"ablation": [ablation]})
        assert getattr(cfg, field) is False
        for other in expected.values():
            if other != field:
                assert getattr(cfg, other) is True
        assert report["ablations"] == [ablation]
