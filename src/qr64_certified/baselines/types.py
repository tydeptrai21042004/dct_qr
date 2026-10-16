from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlindnessTier = Literal["blind", "key_assisted_blind", "semi_blind", "non_blind"]
EngineType = Literal["functional", "class"]

@dataclass(frozen=True)
class BaselineSpec:
    canonical_id: str
    native_id: str
    display_name: str
    domain: str
    algorithm: str
    citation: str
    blindness_tier: BlindnessTier
    engine: EngineType
    fidelity_tier: str
    requires_original_host: bool
    common_4096_payload: bool
    primary_blind_eligible: bool
    disclosure: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    @property
    def method_id(self) -> str:
        """Backward-compatible alias for earlier registry consumers."""
        return self.canonical_id

@dataclass
class IntegratedBaselineKey:
    """Canonical wrapper around an implementation-native extraction key."""
    canonical_id: str
    native_id: str
    engine: EngineType
    native_key: Any
    constructor_params: dict[str, Any] = field(default_factory=dict)

    @property
    def method_id(self) -> str:
        return self.canonical_id

__all__ = ["BaselineSpec", "IntegratedBaselineKey", "BlindnessTier", "EngineType"]
