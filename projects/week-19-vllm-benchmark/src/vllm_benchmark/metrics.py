"""Seconds throughout. Observation timestamps are not claims about kernel-only time."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class RequestTrace:
    submitted_s: float
    token_times_s: tuple[float, ...]
    completed_s: float

    def __post_init__(self) -> None:
        times = tuple(self.token_times_s)
        object.__setattr__(self, "token_times_s", times)
        ordered = (self.submitted_s, *times, self.completed_s)
        if not all(math.isfinite(t) for t in ordered) or any(b < a for a, b in pairwise(ordered)):
            raise ValueError("timestamps must be finite and nondecreasing")

    @property
    def ttft_s(self) -> float | None:
        return self.token_times_s[0] - self.submitted_s if self.token_times_s else None

    @property
    def tpot_s(self) -> float | None:
        n = len(self.token_times_s)
        return (self.token_times_s[-1] - self.token_times_s[0]) / (n - 1) if n > 1 else None

    @property
    def latency_s(self) -> float:
        return self.completed_s - self.submitted_s


def aggregate_metrics(
    traces: Sequence[RequestTrace], *, window_start_s: float, window_end_s: float
) -> dict[str, int | float | None]:
    """All traces must be complete inside an explicit wall window, including idle time.

    Throughput counts ALL output tokens, including each first token. TPOT excludes
    each first token; empty/one-token responses have no decode interval, not TPOT=0.
    Empty completed responses count in requests/s. Failures must be reported separately.
    """
    if not (math.isfinite(window_start_s) and math.isfinite(window_end_s)):
        raise ValueError("window must be finite")
    duration = window_end_s - window_start_s
    if duration <= 0:
        raise ValueError("window must have positive duration")
    if any(t.submitted_s < window_start_s or t.completed_s > window_end_s for t in traces):
        raise ValueError("window must contain every request")
    ttfts = [t.ttft_s for t in traces if t.ttft_s is not None]
    tpots = [t.tpot_s for t in traces if t.tpot_s is not None]
    latencies = [t.latency_s for t in traces]
    intervals = sum(max(0, len(t.token_times_s) - 1) for t in traces)
    decode_time = sum(
        t.token_times_s[-1] - t.token_times_s[0] for t in traces if len(t.token_times_s) > 1
    )
    output_tokens = sum(len(t.token_times_s) for t in traces)
    return {
        "window_s": duration,
        "completed_requests": len(traces),
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / duration,
        "completed_requests_per_s": len(traces) / duration,
        "ttft_samples": len(ttfts),
        "decode_intervals": intervals,
        "mean_ttft_s": float(np.mean(ttfts)) if ttfts else None,
        "p50_ttft_s": float(np.percentile(ttfts, 50)) if ttfts else None,
        "p95_ttft_s": float(np.percentile(ttfts, 95)) if ttfts else None,
        "weighted_tpot_s": decode_time / intervals if intervals else None,
        "mean_request_tpot_s": float(np.mean(tpots)) if tpots else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else None,
    }
