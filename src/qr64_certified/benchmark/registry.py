"""Benchmark selection: corrected theory plus the original 16 baselines."""
from qr64_certified.baselines.registry import (
    ALL_BASELINE_IDS, PRIMARY_BLIND_BASELINE_IDS, get_baseline_spec, normalize_baseline_id)
from qr64_certified.proposals.proposal_registry import (DCT_QR_THEORY,
    SUPPORTED_PROPOSAL_METHODS, normalize_proposal_method_id)
from .types import BenchmarkMethodSpec

PROPOSAL_IDS = (DCT_QR_THEORY,)

def proposal_spec(method_id):
    canonical = normalize_proposal_method_id(method_id)
    return BenchmarkMethodSpec(
        method_id=canonical, method_kind="proposal",
        display_name=SUPPORTED_PROPOSAL_METHODS[canonical]["display_name"],
        blindness_tier="key_assisted_blind", fidelity_tier="proposal",
        requires_original_host=False, common_4096_payload=True,
        payload_size=64, cover_dependent_key=True,
        comparison_group="proposal_key_assisted_4096",
        config_path="configs/dct_qr_theory_after_abc.json")

def baseline_spec(method_id):
    native = get_baseline_spec(normalize_baseline_id(method_id))
    size = 64 if native.common_4096_payload else 16
    return BenchmarkMethodSpec(method_id=native.canonical_id,
        method_kind="baseline", display_name=native.display_name,
        blindness_tier=native.blindness_tier, fidelity_tier=native.fidelity_tier,
        requires_original_host=native.requires_original_host,
        common_4096_payload=native.common_4096_payload, payload_size=size,
        cover_dependent_key=native.blindness_tier != "blind",
        comparison_group=f"baseline_{native.blindness_tier}_{size}x{size}")

def get_method_spec(method_id):
    if str(method_id).strip().lower() == DCT_QR_THEORY:
        return proposal_spec(method_id)
    return baseline_spec(method_id)

def all_method_specs():
    return (proposal_spec(DCT_QR_THEORY),) + tuple(baseline_spec(x) for x in ALL_BASELINE_IDS)

def _ids(value):
    selector = str(value).strip().lower()
    if selector in ("all", "every_method"):
        return PROPOSAL_IDS + tuple(ALL_BASELINE_IDS)
    if selector in ("paper", "paper_comparison", "final", "primary"):
        return PROPOSAL_IDS + tuple(PRIMARY_BLIND_BASELINE_IDS)
    if selector in ("proposal", "proposals", "theory"):
        return PROPOSAL_IDS
    if selector in ("baseline", "baselines"):
        return tuple(ALL_BASELINE_IDS)
    if selector in ("primary_blind", "primary_blind_baselines"):
        return tuple(PRIMARY_BLIND_BASELINE_IDS)
    if selector in ("common_4096", "payload_4096"):
        return tuple(spec.method_id for spec in all_method_specs() if spec.common_4096_payload)
    if selector in ("blind", "strict_blind", "key_assisted_blind", "semi_blind", "non_blind"):
        tier = "blind" if selector == "strict_blind" else selector
        return tuple(spec.method_id for spec in all_method_specs() if spec.blindness_tier == tier)
    return tuple(x.strip() for x in selector.split(",") if x.strip())

def resolve_methods(value):
    raw = _ids(value) if isinstance(value, str) else tuple(value)
    unique = {}
    for x in raw:
        spec = get_method_spec(x)
        unique[spec.method_id] = spec
    if not unique: raise ValueError("No benchmark methods selected")
    return tuple(unique.values())
