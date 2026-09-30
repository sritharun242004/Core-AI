"""Week 19 offline inference contracts. Optional engines are never imported here."""

from .cache import CacheIdentity, PagedKVCache, PrefixCache, prefix_key
from .metrics import RequestTrace, aggregate_metrics
from .quantization import QuantizedWeights, dequantize, quantize
from .speculation import (
    SpeculativeResult,
    acceptance_probability,
    residual_distribution,
    speculative_sample,
)

__all__ = [
    "CacheIdentity",
    "PagedKVCache",
    "PrefixCache",
    "QuantizedWeights",
    "RequestTrace",
    "SpeculativeResult",
    "acceptance_probability",
    "aggregate_metrics",
    "dequantize",
    "prefix_key",
    "quantize",
    "residual_distribution",
    "speculative_sample",
]
