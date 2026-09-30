"""Hypothetical token accounting, eligibility routing and an isolated bounded cache."""

import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass


def finite_nonnegative(value: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in asdict(self).values()):
            raise ValueError("token counts must be nonnegative integers")
        if self.cached_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("cached and write input partitions cannot overlap or exceed input")


@dataclass(frozen=True)
class Price:
    input_per_million: float
    cached_per_million: float
    cache_write_per_million: float
    output_per_million: float

    def __post_init__(self):
        if any(not finite_nonnegative(p) for p in asdict(self).values()):
            raise ValueError("prices must be finite and nonnegative")


def estimate_cost(usage: TokenUsage, price: Price, *, batch_factor: float = 1) -> float:
    """Writes are a disjoint input category at an all-in write price, not a surcharge.

    The illustrative batch multiplier applies to every category here. Real provider
    discounts and cache eligibility differ; configure from a dated rate card.
    """
    if not finite_nonnegative(batch_factor) or not 0 < batch_factor <= 1:
        raise ValueError("batch_factor must lie in (0, 1]")
    fresh = usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens
    return (
        batch_factor
        * (
            fresh * price.input_per_million
            + usage.cached_tokens * price.cached_per_million
            + usage.cache_write_tokens * price.cache_write_per_million
            + usage.output_tokens * price.output_per_million
        )
        / 1_000_000
    )


@dataclass(frozen=True)
class ModelOption:
    name: str
    price: Price
    quality: float
    latency_ms: float

    def __post_init__(self):
        if (
            not self.name
            or not finite_nonnegative(self.quality)
            or self.quality > 1
            or (not finite_nonnegative(self.latency_ms))
        ):
            raise ValueError("invalid model option")


def route_model(
    options: list[ModelOption],
    usage: TokenUsage,
    *,
    min_quality: float,
    max_latency_ms: float,
    max_cost: float,
) -> ModelOption:
    if (
        not finite_nonnegative(min_quality)
        or min_quality > 1
        or any(not finite_nonnegative(v) for v in (max_latency_ms, max_cost))
    ):
        raise ValueError("invalid routing constraints")
    eligible = [
        option
        for option in options
        if option.quality >= min_quality
        and option.latency_ms <= max_latency_ms
        and estimate_cost(usage, option.price) <= max_cost
    ]
    if not eligible:
        raise ValueError("no eligible model; do not silently relax constraints")
    return min(eligible, key=lambda option: (estimate_cost(usage, option.price), option.name))


@dataclass(frozen=True)
class CacheKey:
    tenant: str
    model: str
    prompt: str
    context_revision: str
    tool_revision: str
    parameters: str

    def digest(self) -> str:
        if any(not isinstance(value, str) or not value for value in asdict(self).values()):
            raise ValueError("all cache key dimensions must be nonempty strings")
        if any(len(value) > 32_768 for value in asdict(self).values()):
            raise ValueError("cache key field too large")
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class ResponseCache:
    """Process-local exact response cache, NOT provider KV/prompt caching.

    Key partitioning supports isolation; the host must supply the authenticated
    tenant/principal scope. A model-provided tenant string is not authorization.
    """

    def __init__(
        self,
        capacity: int = 32,
        ttl_seconds: float = 60,
        max_bytes: int = 65_536,
        clock: Callable[[], float] = time.monotonic,
    ):
        if (
            type(capacity) is not int
            or not 1 <= capacity <= 10_000
            or (type(max_bytes) is not int or not 1 <= max_bytes <= 1_000_000)
            or not finite_nonnegative(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("invalid cache limits")
        self.capacity, self.ttl_seconds, self.max_bytes = capacity, ttl_seconds, max_bytes
        self.clock = clock
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: CacheKey) -> dict | None:
        digest = key.digest()
        entry = self._items.get(digest)
        if entry is None:
            return None
        expiry, value = entry
        if self.clock() >= expiry:
            del self._items[digest]
            return None
        self._items.move_to_end(digest)
        return deepcopy(value)

    def put(self, key: CacheKey, value: dict, *, success: bool = True) -> None:
        if not success:
            return
        digest = key.digest()
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded.encode()) > self.max_bytes:
            raise ValueError("response exceeds cache entry bound")
        self._items[digest] = (self.clock() + self.ttl_seconds, json.loads(encoded))
        self._items.move_to_end(digest)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
