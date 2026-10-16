"""Unified deterministic attack library for proposals and baselines."""

from .operations import apply_attack, available_attack_groups
from .presets import ATTACK_SUITES, get_attack_suite, list_attack_suites
from .types import AttackConfig

__all__ = [
    "AttackConfig", "apply_attack", "available_attack_groups",
    "ATTACK_SUITES", "get_attack_suite", "list_attack_suites",
]
