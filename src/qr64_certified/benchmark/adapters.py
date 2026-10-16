"""Minimal, mathematically unchanged adapter for corrected theory and baselines."""
import json
from dataclasses import replace
from pathlib import Path
from qr64_certified.baselines.registry import embed_baseline, extract_baseline
from qr64_certified.proposals.dct_qr_theory import DCTQRTheoryConfig
from qr64_certified.proposals.proposal_registry import (
    DCT_QR_THEORY, embed_proposal, extract_proposal)
from .types import BaselineRunParameters, BenchmarkMethodSpec
import numpy as np

def load_baseline_parameters(path):
    if path is None or not Path(path).exists(): return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: BaselineRunParameters(repeat=value.get("repeat", 1),
             step=value.get("step"), method_params=dict(value.get("method_params", {})))
             for key, value in payload.items()}

class BenchmarkAdapter:
    def __init__(self, spec: BenchmarkMethodSpec, root: Path, baseline_parameters=None):
        self.spec = spec
        self.root = Path(root)
        self.baseline_parameters = baseline_parameters or BaselineRunParameters()
        if spec.method_kind == "proposal":
            if spec.method_id != DCT_QR_THEORY:
                raise ValueError("Only the corrected theory proposal is supported")
            config_path = self.root / spec.config_path
            raw = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
            self._proposal_config = DCTQRTheoryConfig.from_mapping(raw)

    def _config(self, seed):
        return replace(self._proposal_config, seed=int(seed)).validated()

    def configuration_summary(self, seed):
        if self.spec.method_kind == "proposal":
            return self._config(seed).to_dict()
        return {"repeat": self.baseline_parameters.repeat,
                "step": self.baseline_parameters.step,
                "method_params": dict(self.baseline_parameters.method_params)}

    def embed(self, host, watermark, *, seed):
        if self.spec.method_kind == "proposal":
            im, key, meta = embed_proposal(self.spec.method_id, host, watermark,
                config=self._config(seed), return_metadata=True)
            return np.asarray(im, dtype=np.uint8), key, dict(meta or {})
        im, key = embed_baseline(self.spec.method_id, host, watermark, seed=int(seed),
            repeat=self.baseline_parameters.repeat, step=self.baseline_parameters.step,
            method_params=self.baseline_parameters.method_params)
        return np.asarray(im, dtype=np.uint8), key, {}

    def extract(self, image, key, *, original_host=None):
        if self.spec.method_kind == "proposal":
            wm, meta = extract_proposal(image, key, return_metadata=True)
            return np.asarray(wm, dtype=np.uint8), dict(meta or {})
        wm = extract_baseline(image, key,
            original_host=original_host if self.spec.requires_original_host else None)
        return np.asarray(wm, dtype=np.uint8), {}
