"""Unified baseline package.

All public baseline IDs follow:
``<domain>_<algorithm>_<citation>_<access>``.
Legacy IDs remain accepted as aliases, but are never emitted in new reports.
"""

from .registry import (
    ALL_BASELINE_IDS,
    PRIMARY_BLIND_BASELINE_IDS,
    SPECS,
    embed_baseline,
    extract_baseline,
    get_baseline_spec,
    list_baselines,
    normalize_baseline_id,
)
from .types import BaselineSpec, IntegratedBaselineKey
from .evaluation import evaluate_baseline_attacks

__all__ = [
    "ALL_BASELINE_IDS", "PRIMARY_BLIND_BASELINE_IDS", "SPECS",
    "BaselineSpec", "IntegratedBaselineKey", "normalize_baseline_id",
    "get_baseline_spec", "list_baselines", "embed_baseline", "extract_baseline",
    "evaluate_baseline_attacks",
]
