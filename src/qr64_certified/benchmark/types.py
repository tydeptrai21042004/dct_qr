from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MethodKind = Literal["proposal", "baseline"]


@dataclass(frozen=True)
class BenchmarkMethodSpec:
    method_id: str
    method_kind: MethodKind
    display_name: str
    blindness_tier: str
    fidelity_tier: str
    requires_original_host: bool
    common_4096_payload: bool
    payload_size: int
    cover_dependent_key: bool
    comparison_group: str
    config_path: str | None = None

    @property
    def payload_bits(self) -> int:
        return int(self.payload_size * self.payload_size)


@dataclass(frozen=True)
class BaselineRunParameters:
    repeat: int | str = 1
    step: float | None = None
    method_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkProtocol:
    protocol_id: str = "quick_unified_v1"
    host_dir: str = "data/host"
    watermarks: tuple[str, ...] = ("data/watermark/wm.png",)
    seeds: tuple[int, ...] = (2026,)
    attack_suite: str = "sanity"
    methods: str | tuple[str, ...] = "paper_comparison"
    output_dir: str = "results/unified_benchmark"
    continue_on_error: bool = True
    resume: bool = True
    strict_clean_proposals: bool = True
    host_limit: int | None = 1
    attack_limit: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BenchmarkProtocol":
        raw = dict(value)
        if "watermark" in raw and "watermarks" not in raw:
            raw["watermarks"] = [raw.pop("watermark")]
        if "watermarks" in raw:
            raw["watermarks"] = tuple(str(v) for v in raw["watermarks"])
        if "seeds" in raw:
            raw["seeds"] = tuple(int(v) for v in raw["seeds"])
        if isinstance(raw.get("methods"), list):
            raw["methods"] = tuple(str(v) for v in raw["methods"])
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown benchmark protocol fields: {', '.join(unknown)}")
        protocol = cls(**raw)
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("protocol_id cannot be empty")
        if not self.watermarks:
            raise ValueError("At least one watermark is required")
        if not self.seeds:
            raise ValueError("At least one seed is required")
        if self.host_limit is not None and self.host_limit <= 0:
            raise ValueError("host_limit must be positive or null")
        if self.attack_limit is not None and self.attack_limit <= 0:
            raise ValueError("attack_limit must be positive or null")

    def resolve_path(self, root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path


__all__ = [
    "MethodKind", "BenchmarkMethodSpec", "BaselineRunParameters", "BenchmarkProtocol",
]
