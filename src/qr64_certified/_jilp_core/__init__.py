"""Internal JILP-QIM core retained solely because qr64_certified builds on it.

It is not registered as an imported baseline or as a separate proposal.
"""
from .config import DEFAULT_CONFIG, JILPQIMConfig

__all__ = ["DEFAULT_CONFIG", "JILPQIMConfig"]
