"""Exactly one proposed watermarking method: corrected DCT-QR theory."""
from . import dct_qr_theory
from .dct_qr_theory import DCTQRTheoryConfig, DCTQRTheoryKey, embed, extract
from .proposal_registry import DCT_QR_THEORY, SUPPORTED_PROPOSAL_METHODS, embed_proposal, extract_proposal, list_supported_methods, normalize_proposal_method_id, default_config_for_method
from .flags import ABLATION_FLAGS, HYPERPARAMETER_FLAGS, apply_proposal_flags
