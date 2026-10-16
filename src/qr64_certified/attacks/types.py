from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class AttackConfig:
    """Deterministic attack configuration used by every method."""
    attack_id: str
    group: str
    params: dict[str, Any] = field(default_factory=dict)
    category: str = "other"
    severity: str = "custom"

    @property
    def name(self) -> str:
        """Backward-compatible alias used by older result writers."""
        return self.attack_id

__all__ = ["AttackConfig"]
