"""Single-threaded ownership simulator and exact, namespace-scoped prefix keys."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .quantization import positive_int


class PagedKVCache:
    """Logical block table only: no tensors, eviction, sharing, or attention kernels."""

    def __init__(self, num_pages: int, page_size: int):
        positive_int(num_pages, "num_pages")
        positive_int(page_size, "page_size")
        self.num_pages = num_pages
        self.page_size = page_size
        self._free = set(range(num_pages))
        self._pages: dict[str, tuple[int, ...]] = {}
        self._tokens: dict[str, int] = {}

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def internal_waste_tokens(self) -> int:
        return sum(len(self._pages[key]) * self.page_size - n for key, n in self._tokens.items())

    def pages_for(self, request: str) -> tuple[int, ...]:
        return self._pages[request]

    def tokens_for(self, request: str) -> int:
        return self._tokens[request]

    def _reserve(self, count: int) -> tuple[int, ...]:
        if count > self.free_pages:
            raise MemoryError("KV page capacity exhausted; request state is unchanged")
        pages = tuple(sorted(self._free)[:count])
        self._free.difference_update(pages)
        return pages

    def allocate(self, request: str, tokens: int) -> tuple[int, ...]:
        positive_int(tokens, "tokens", allow_zero=True)
        if not isinstance(request, str) or not request or request in self._pages:
            raise ValueError("request must be a nonempty, unique ID")
        pages = self._reserve((tokens + self.page_size - 1) // self.page_size)
        self._pages[request] = pages
        self._tokens[request] = tokens
        return pages

    def append(self, request: str, tokens: int) -> tuple[int, ...]:
        positive_int(tokens, "tokens", allow_zero=True)
        new_tokens = self._tokens[request] + tokens
        needed = (new_tokens + self.page_size - 1) // self.page_size
        extra = self._reserve(needed - len(self._pages[request]))
        self._pages[request] += extra
        self._tokens[request] = new_tokens
        return self._pages[request]

    def release(self, request: str) -> None:
        pages = self._pages.pop(request)  # Unknown/double-free fails before changing capacity.
        self._free.update(pages)
        del self._tokens[request]

    def check_invariants(self) -> None:
        pages = [page for block_table in self._pages.values() for page in block_table]
        assert len(pages) == len(set(pages)), "page aliasing"
        assert not (set(pages) & self._free), "allocated page on free list"
        assert set(pages) | self._free == set(range(self.num_pages)), "lost/out-of-range page"
        assert self._pages.keys() == self._tokens.keys()
        for request, tokens in self._tokens.items():
            assert len(self._pages[request]) == (tokens + self.page_size - 1) // self.page_size


@dataclass(frozen=True)
class CacheIdentity:
    """Versioned computational state plus tenant boundary (not authentication)."""

    tenant: str
    model_revision: str
    tokenizer_revision: str
    adapter_revision: str
    position_config: str
    kv_dtype: str


def prefix_key(identity: CacheIdentity, token_ids: Sequence[int]) -> str:
    """Exact token sequence, never decoded text or ambiguous string concatenation."""
    fields = asdict(identity)
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise ValueError("every cache identity field must be a nonempty string")
    tokens = list(token_ids)
    for token in tokens:
        positive_int(token, "token ID", allow_zero=True)
    payload = json.dumps(
        {"version": 1, "identity": fields, "tokens": tokens}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class PrefixCache:
    """Bounded exact-entry LRU; values stand in for KV handles, not real GPU blocks.

    Caller owns authentication, TTL, memory budget, lifecycle, and reference counts.
    This cache never searches for a shorter matching prefix automatically.
    """

    def __init__(self, max_entries: int):
        positive_int(max_entries, "max_entries")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, object] = OrderedDict()

    def put(self, identity: CacheIdentity, token_ids: Sequence[int], value: object) -> None:
        if value is None:
            raise ValueError("None is reserved for cache misses")
        key = prefix_key(identity, token_ids)
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, identity: CacheIdentity, token_ids: Sequence[int]) -> object | None:
        key = prefix_key(identity, token_ids)
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
