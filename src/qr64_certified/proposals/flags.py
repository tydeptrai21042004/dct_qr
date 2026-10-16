"""Six reproducible theory-only one-factor ablation switches."""
from dataclasses import replace
from typing import Any, Mapping
from .dct_qr_theory import DCTQRTheoryConfig
from .proposal_registry import DCT_QR_THEORY, normalize_proposal_method_id

ABLATION_TO_FIELD = {
    "no_opponent_term": "use_opponent_term",
    "uniform_step": "use_adaptive_beta_steps",
    "no_global_coset": "use_global_coset",
    "no_gain_normalization": "use_gain_normalization",
    "no_spatial_icm": "use_spatial_icm",
    "no_sync_search": "use_sync_search",
}
ABLATION_FLAGS = {DCT_QR_THEORY: tuple(ABLATION_TO_FIELD)}
HYPERPARAMETER_FLAGS = {DCT_QR_THEORY: tuple(DCTQRTheoryConfig.__dataclass_fields__)}

def apply_proposal_flags(method_id: str, config, flags: Mapping[str, Any]):
    normalize_proposal_method_id(method_id)
    cfg = DCTQRTheoryConfig.from_mapping(config)
    names = flags.get("ablation", ())
    if isinstance(names, str): names = (names,)
    invalid = set(names) - set(ABLATION_TO_FIELD)
    if invalid: raise ValueError(f"Unknown corrected-theory ablations: {sorted(invalid)}")
    for name in names:
        cfg = replace(cfg, **{ABLATION_TO_FIELD[name]: False})
    overrides = dict(flags.get("hyperparameters", {}))
    invalid_overrides = set(overrides) - set(DCTQRTheoryConfig.__dataclass_fields__)
    if invalid_overrides: raise ValueError(f"Unknown theory parameters: {sorted(invalid_overrides)}")
    cfg = replace(cfg, **overrides).validated()
    return cfg, {"ablations": list(names), "hyperparameters": overrides}
