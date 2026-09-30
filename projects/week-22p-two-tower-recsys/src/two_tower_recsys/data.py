"""Positive-only implicit feedback, with explicit train/holdout boundaries."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Interaction:
    user: int
    item: int
    timestamp: int

    def __post_init__(self):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in (self.user, self.item, self.timestamp)
        ):
            raise ValueError("user, item, timestamp must be nonnegative integers")


def synthetic_movielens(seed: int = 22) -> list[Interaction]:
    """24 users, 18 items, three planted tastes; not actual MovieLens records."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(24):
        items = rng.permutation(np.arange(6) + 6 * (user % 3))[:5]
        rows.extend(Interaction(user, int(item), t + 1) for t, item in enumerate(items))
    return rows


def temporal_split(rows: list[Interaction], holdout: int = 1):
    """Leave the last n unique edges per user out, rejecting ambiguous time ties.

    This is per-user chronology, NOT a global deployment-time cutoff. Repeated
    user/item events must be aggregated explicitly before calling this function.
    """
    if not isinstance(holdout, int) or holdout < 1:
        raise ValueError("holdout must be a positive integer")
    if len({(r.user, r.item) for r in rows}) != len(rows):
        raise ValueError("duplicate user-item edges must be aggregated first")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.user].append(row)
    fit, test = [], []
    for user in sorted(grouped):
        ordered = sorted(grouped[user], key=lambda r: (r.timestamp, r.item))
        if len(ordered) <= holdout:
            raise ValueError("each user needs training and holdout interactions")
        if ordered[-holdout - 1].timestamp >= ordered[-holdout].timestamp:
            raise ValueError("timestamp ties at split boundary are ambiguous")
        fit.extend(ordered[:-holdout])
        test.extend(ordered[-holdout:])
    return fit, test


def load_movielens(path: str | Path, min_rating: float = 4.0):
    """Parse an EXISTING MovieLens ratings.csv; never fetch anything.

    Return positive interactions plus sorted raw-ID -> contiguous-ID maps.
    Only the modern CSV schema is supported (not legacy ratings.dat/u.data).
    Repeated positives are retained so temporal_split can reject silent leakage.
    """
    if not np.isfinite(min_rating):
        raise ValueError("min_rating must be finite")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"userId", "movieId", "rating", "timestamp"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("expected MovieLens ratings.csv columns")
        raw = []
        for row in reader:
            rating = float(row["rating"])
            if not np.isfinite(rating):
                raise ValueError("ratings must be finite")
            if rating >= min_rating:
                raw.append((int(row["userId"]), int(row["movieId"]), int(row["timestamp"])))
    users = {u: i for i, u in enumerate(sorted({r[0] for r in raw}))}
    items = {v: i for i, v in enumerate(sorted({r[1] for r in raw}))}
    return [Interaction(users[u], items[v], t) for u, v, t in raw], users, items
