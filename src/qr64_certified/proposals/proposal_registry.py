"""Registry containing exactly one proposed method: corrected dct_qr_theory."""
from typing import Any
import numpy as np
from .dct_qr_theory import DCTQRTheoryConfig, DCTQRTheoryKey, embed, extract

DCT_QR_THEORY = "dct_qr_theory"
SUPPORTED_PROPOSAL_METHODS = {DCT_QR_THEORY: {
    "id": DCT_QR_THEORY,
    "display_name": "Corrected DCT-QR Theory-Grounded Adaptive QIM",
    "scientific_status": "theory-grounded; host-blind/key-assisted extraction",
}}

def normalize_proposal_method_id(method_id: str) -> str:
    if str(method_id).strip().lower() != DCT_QR_THEORY:
        raise KeyError("Only dct_qr_theory is a supported proposal; legacy dct_qr is removed")
    return DCT_QR_THEORY

def list_supported_methods():
    return list(SUPPORTED_PROPOSAL_METHODS.values())

def default_config_for_method(method_id: str):
    normalize_proposal_method_id(method_id)
    return DCTQRTheoryConfig()

def embed_proposal(method_id: str, host: np.ndarray, watermark: np.ndarray, *, config=None, return_metadata=False):
    normalize_proposal_method_id(method_id)
    return embed(host, watermark, config=DCTQRTheoryConfig.from_mapping(config), return_metadata=return_metadata)

def extract_proposal(image: np.ndarray, key: DCTQRTheoryKey, *, return_metadata=False):
    if not isinstance(key, DCTQRTheoryKey):
        raise TypeError("Extraction requires a corrected DCTQRTheoryKey")
    return extract(image, key, return_metadata=return_metadata)
