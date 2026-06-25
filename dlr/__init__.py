"""Duplicated Latent Residuals for low-rank pre-training."""

from .layers import DLRLinear, expand_latent, expansion_index
from .modeling import DLRConfig, DLRModel

__all__ = [
    "DLRConfig",
    "DLRLinear",
    "DLRModel",
    "expand_latent",
    "expansion_index",
]
