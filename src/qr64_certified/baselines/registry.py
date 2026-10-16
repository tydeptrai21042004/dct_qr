from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from .implementations.dct import (
    ca_qim_a2_mao2024,
    dct_dm_qim_chen2001,
    dct_iss_malvar2003,
    dct_spread_spectrum_cox1997,
    dct_stdm_qim_chen2001,
    dew_langelaar2001,
)
from .implementations.transform.dwt_hd_svd2025 import DWTHDSVD2025
from .implementations.transform.dwt_wht_svd2024 import DWTWHTSVD2024
from .implementations.transform.gaata2022_dwt_hess_fwa import Gaata2022DWTHessFWA
from .implementations.transform.guo2017_dwt_qr_fa import Guo2017DWTQRFA
from .implementations.transform.hess_nha2023 import HessNha2023Hessenberg
from .implementations.transform.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from .implementations.transform.qwt_qsvd_zhang2022 import QWTQSVDZhang2022
from .implementations.transform.roy2018_dwt_svd import Roy2018DWTSVD
from .implementations.transform.zhu2021_iwt_svd import Zhu2021IWTSVD
from .metadata import BASELINE_METADATA
from .types import BaselineSpec, IntegratedBaselineKey

# Functional implementations preserve their original equations and native key
# schemas. Canonical naming is handled only by this unified registry.
_FUNCTIONAL_MODULES = {
    "dct_dm_qim_chen2001": dct_dm_qim_chen2001,
    "dct_stdm_qim_chen2001": dct_stdm_qim_chen2001,
    "dct_iss_malvar2003": dct_iss_malvar2003,
    "ca_qim_a2_mao2024": ca_qim_a2_mao2024,
    "dct_spread_spectrum_cox1997": dct_spread_spectrum_cox1997,
    "dew_langelaar2001": dew_langelaar2001,
}


def _class_factory(native_id: str, params: dict[str, Any]):
    options = dict(params)
    options.setdefault("mode", "adapt")
    if native_id == "kumar2021":
        return Kumar2021DWTEntropy(**options)
    if native_id == "guo2017_dwt_qr_fa":
        return Guo2017DWTQRFA(**options)
    if native_id == "gaata2022_dwt_hess_fwa":
        return Gaata2022DWTHessFWA(**options)
    if native_id == "dwt_hd_svd_2025":
        return DWTHDSVD2025(**options)
    if native_id == "hess_nha2023":
        return HessNha2023Hessenberg(**options)
    if native_id == "roy2018_dwt_svd":
        return Roy2018DWTSVD(**options)
    if native_id == "dwt_wht_svd_2024":
        return DWTWHTSVD2024(**options)
    if native_id == "qwt_qsvd_zhang2022_blind":
        options["extraction_mode"] = "blind"
        return QWTQSVDZhang2022(**options)
    if native_id == "qwt_qsvd_zhang2022_semiblind":
        options["extraction_mode"] = "semi-blind"
        return QWTQSVDZhang2022(**options)
    if native_id == "zhu2021_iwt_svd_adapted":
        return Zhu2021IWTSVD(**options)
    raise KeyError(f"No class implementation registered for native id: {native_id}")


# Short aliases and all pre-reorganization IDs remain accepted. New reports and
# result files always emit the canonical ID.
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "dct_dm_qim_chen2001_blind": ("dm_qim",),
    "dct_stdm_qim_chen2001_blind": ("stdm", "stdm_qim"),
    "dct_iss_malvar2003_blind": ("iss",),
    "dct_ca_qim_mao2024_semiblind": ("ca_qim",),
    "dct_spread_spectrum_cox1997_nonblind": ("cox", "cox_ss"),
    "dct_dew_langelaar2001_blind": ("dew", "differential_energy"),
    "dwt_entropy_kumar2021_nonblind": ("kumar2021",),
    "dwt_qr_fa_guo2017_keyassisted": ("guo2017",),
    "dwt_hessenberg_fwa_gaata2022_keyassisted": ("gaata2022",),
    "dwt_hessenberg_svd_paper2025_semiblind": ("dwt_hd_svd_2025",),
    "hessenberg_nha2023_blind": ("hess_nha2023",),
    "dwt_svd_roy2018_semiblind": ("roy2018",),
    "dwt_wht_svd_kumar2024_semiblind": ("dwt_wht_svd_2024",),
    "qwt_qsvd_zhang2022_blind": (),
    "qwt_qsvd_zhang2022_semiblind": (),
    "iwt_svd_qim_zhu2021_blind": ("zhu2021",),
}

SPECS: dict[str, BaselineSpec] = {}
for canonical_id, row in BASELINE_METADATA.items():
    aliases = tuple(dict.fromkeys((row["native_id"],) + _EXTRA_ALIASES.get(canonical_id, ())))
    SPECS[canonical_id] = BaselineSpec(
        canonical_id=canonical_id,
        native_id=row["native_id"],
        display_name=row["display_name"],
        domain=row["domain"],
        algorithm=row["algorithm"],
        citation=row["citation"],
        blindness_tier=row["blindness_tier"],
        engine=row["engine"],
        fidelity_tier=row["fidelity_tier"],
        requires_original_host=row["requires_original_host"],
        common_4096_payload=row["common_4096_payload"],
        primary_blind_eligible=row["primary_blind_eligible"],
        disclosure=dict(row["disclosure"]),
        aliases=aliases,
    )

ALIASES: dict[str, str] = {}
for canonical_id, spec in SPECS.items():
    ALIASES[canonical_id] = canonical_id
    for alias in spec.aliases:
        ALIASES[str(alias).lower()] = canonical_id

ALL_BASELINE_IDS = tuple(SPECS)
PRIMARY_BLIND_BASELINE_IDS = tuple(
    method_id for method_id, spec in SPECS.items() if spec.primary_blind_eligible
)


def normalize_baseline_id(method_id: str) -> str:
    normalized = str(method_id).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = ALIASES.get(normalized)
    if canonical is None:
        raise KeyError(
            f"Unknown baseline '{method_id}'. Valid canonical IDs: {', '.join(ALL_BASELINE_IDS)}"
        )
    return canonical


def get_baseline_spec(method_id: str) -> BaselineSpec:
    return SPECS[normalize_baseline_id(method_id)]


def list_baselines(
    *,
    blindness_tier: str | None = None,
    domain: str | None = None,
    primary_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS.values():
        if blindness_tier is not None and spec.blindness_tier != blindness_tier:
            continue
        if domain is not None and spec.domain != domain:
            continue
        if primary_only and not spec.primary_blind_eligible:
            continue
        rows.append(asdict(spec))
    return rows


def _normalize_inputs(host_rgb: np.ndarray, watermark_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    host = np.asarray(host_rgb, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] != 3:
        raise ValueError(f"Host must be RGB HxWx3, got {host.shape}")
    raw_wm = np.asarray(watermark_binary)
    threshold = 0 if raw_wm.size and float(np.max(raw_wm)) <= 1.0 else 127
    wm = np.where(raw_wm > threshold, 255, 0).astype(np.uint8)
    if wm.ndim != 2:
        raise ValueError(f"Watermark must be a 2D binary array, got {wm.shape}")
    return host, wm


def _call_with_optional_params(
    fn: Callable[..., Any],
    *args: Any,
    method_params: dict[str, Any] | None = None,
    **kwargs: Any,
):
    try:
        return fn(*args, method_params=method_params, **kwargs)
    except TypeError as exc:
        if "method_params" not in str(exc):
            raise
        return fn(*args, **kwargs)


def embed_baseline(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, IntegratedBaselineKey]:
    canonical = normalize_baseline_id(method_id)
    spec = SPECS[canonical]
    host, wm = _normalize_inputs(host_rgb, watermark_binary)
    params = dict(method_params or {})

    if spec.engine == "functional":
        module = _FUNCTIONAL_MODULES[spec.native_id]
        watermarked, native_key = _call_with_optional_params(
            module.embed,
            host,
            wm,
            seed=seed,
            repeat=repeat,
            step=step,
            method_params=params,
        )
    else:
        method = _class_factory(spec.native_id, params)
        watermarked, native_key = method.embed(host, wm)

    return watermarked, IntegratedBaselineKey(
        canonical_id=canonical,
        native_id=spec.native_id,
        engine=spec.engine,
        native_key=native_key,
        constructor_params=params,
    )


def extract_baseline(
    image_rgb: np.ndarray,
    key: IntegratedBaselineKey,
    *,
    original_host: np.ndarray | None = None,
) -> np.ndarray:
    canonical = normalize_baseline_id(key.canonical_id)
    spec = SPECS[canonical]
    image = np.asarray(image_rgb, dtype=np.uint8)
    if key.native_id != spec.native_id or key.engine != spec.engine:
        raise ValueError(
            f"Key implementation mismatch for {canonical}: expected "
            f"{spec.native_id}/{spec.engine}, got {key.native_id}/{key.engine}"
        )

    if spec.engine == "functional":
        return _FUNCTIONAL_MODULES[spec.native_id].extract(image, key.native_key)

    if spec.requires_original_host and original_host is None:
        raise ValueError(f"Baseline {canonical} requires original_host during extraction")
    method = _class_factory(spec.native_id, dict(key.constructor_params))
    return method.extract(image, key.native_key, host_rgb=original_host)


__all__ = [
    "SPECS", "ALIASES", "ALL_BASELINE_IDS", "PRIMARY_BLIND_BASELINE_IDS",
    "normalize_baseline_id", "get_baseline_spec", "list_baselines",
    "embed_baseline", "extract_baseline",
]
