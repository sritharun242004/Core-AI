"""Expanding-window rolling origins and actual tiny residual backcast blocks."""

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

type SeriesArray = NDArray[np.float64]
type ForecastArray = NDArray[np.float32]


class ForecastReport(TypedDict):
    origin: int
    forecast: ForecastArray
    baseline: SeriesArray
    target: SeriesArray
    mae: float
    baseline_mae: float
    mean: float
    scale: float
    initial_loss: float
    final_loss: float


class _TorchRandom(Protocol):
    def fork_rng(self) -> AbstractContextManager[None]: ...


class _TorchApi(Protocol):
    random: _TorchRandom

    def manual_seed(self, seed: int) -> torch.Generator: ...


class _Backward(Protocol):
    def backward(self) -> None: ...


class _Steppable(Protocol):
    def step(self) -> None: ...


def _series(values: ArrayLike) -> SeriesArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("series must be a nonempty finite one-dimensional array")
    return result


def forecast_fixture(length: int = 100) -> SeriesArray:
    if length < 1:
        raise ValueError("length must be positive")
    t = np.arange(length)
    return 10 + 0.04 * t + 2 * np.sin(2 * np.pi * t / 6) + 0.2 * np.cos(2 * np.pi * t / 3)


def training_windows(
    prefix: ArrayLike, *, lookback: int, horizon: int
) -> tuple[SeriesArray, SeriesArray]:
    values = _series(prefix)
    if min(lookback, horizon) < 1 or len(values) < lookback + horizon:
        raise ValueError("prefix must fit at least one complete input/target window")
    starts = range(lookback, len(values) - horizon + 1)
    return np.stack([values[t - lookback : t] for t in starts]), np.stack(
        [values[t : t + horizon] for t in starts]
    )


def rolling_origins(length: int, *, initial: int, horizon: int, step: int) -> list[int]:
    if min(initial, horizon, step) < 1 or initial + horizon > length:
        raise ValueError("positive geometry and at least one complete forecast required")
    return list(range(initial, length - horizon + 1, step))


def seasonal_naive(history: ArrayLike, *, horizon: int, season: int) -> SeriesArray:
    values = _series(history)
    if min(horizon, season) < 1 or len(values) < season:
        raise ValueError("need a complete past season and a positive horizon")
    return np.resize(values[-season:], horizon).copy()


class ResidualBlock(nn.Module):
    """Generic learned backcast/forecast heads; no fixed trend/seasonal basis."""

    def __init__(self, lookback: int, horizon: int, width: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(lookback, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.backcast = nn.Linear(width, lookback)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.hidden(residual)
        return self.backcast(hidden), self.forecast(hidden)


class TinyNBeats(nn.Module):
    """N-BEATS-like generic residual stack, not the full published architecture.

    x_next = x - backcast; y_hat = sum(forecasts). The last backcast is unused
    by forecast loss because there is no downstream residual block.
    """

    def __init__(
        self, lookback: int, horizon: int, *, width: int = 32, blocks: int = 2, seed: int = 23
    ) -> None:
        super().__init__()
        if min(lookback, horizon, width, blocks) < 1:
            raise ValueError("positive architecture dimensions required")
        self.lookback, self.horizon = lookback, horizon
        # Narrow the partially annotated third-party RNG call signatures.
        torch_api = cast(_TorchApi, torch)
        with torch_api.random.fork_rng():
            torch_api.manual_seed(seed)
            self.blocks = nn.ModuleList(
                [ResidualBlock(lookback, horizon, width) for _ in range(blocks)]
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.lookback:
            raise ValueError("expected [batch, lookback] inputs")
        residual = x
        prediction = x.new_zeros((len(x), self.horizon))
        # ModuleList erases element types; construction above installs only ResidualBlocks.
        for block in cast(Iterable[ResidualBlock], self.blocks):
            backcast, forecast = block(residual)
            residual = residual - backcast
            prediction = prediction + forecast
        return prediction


def train_forecaster(
    prefix: ArrayLike,
    *,
    lookback: int = 12,
    horizon: int = 4,
    steps: int = 100,
    lr: float = 0.01,
    seed: int = 23,
) -> tuple[TinyNBeats, list[float], float, float]:
    """Fit preprocessing AND weights on prefix only; no validation/test tuning."""
    values = _series(prefix)
    if steps < 1 or lr <= 0:
        raise ValueError("positive steps and learning rate required")
    mean = float(values.mean())
    scale = float(values.std())
    if scale < 1e-8:
        scale = 1.0
    windows, targets = training_windows((values - mean) / scale, lookback=lookback, horizon=horizon)
    x, y = torch.tensor(windows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    model = TinyNBeats(lookback, horizon, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for step in range(steps + 1):
        loss = (model(x) - y).square().mean()
        history.append(float(loss.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        cast(_Backward, loss).backward()
        cast(_Steppable, optimizer).step()
    model.eval()
    return model, history, mean, scale


def rolling_forecast(
    series: ArrayLike,
    *,
    initial: int = 60,
    horizon: int = 4,
    step: int = 4,
    lookback: int = 12,
    season: int = 6,
    steps: int = 100,
    seed: int = 23,
) -> list[ForecastReport]:
    """Refit independently at each origin; targets start at origin, not origin+1.

    Returns raw forecast/target arrays and errors, not an invalid iid window CI.
    Overlapping horizons create dependent errors even with a correct time split.
    """
    values = _series(series)
    reports: list[ForecastReport] = []
    for origin in rolling_origins(len(values), initial=initial, horizon=horizon, step=step):
        prefix = values[:origin]
        model, history, mean, scale = train_forecaster(
            prefix, lookback=lookback, horizon=horizon, steps=steps, seed=seed
        )
        inputs = torch.tensor(((prefix[-lookback:] - mean) / scale)[None], dtype=torch.float32)
        with torch.no_grad():
            # This CPU model and its inputs use float32; Tensor.numpy loses that dtype.
            forecast = cast(ForecastArray, model(inputs)[0].numpy()) * scale + mean
        baseline = seasonal_naive(prefix, horizon=horizon, season=season)
        target = values[origin : origin + horizon].copy()
        reports.append(
            {
                "origin": origin,
                "forecast": forecast,
                "baseline": baseline,
                "target": target,
                "mae": float(np.abs(forecast - target).mean()),
                "baseline_mae": float(np.abs(baseline - target).mean()),
                "mean": mean,
                "scale": scale,
                "initial_loss": history[0],
                "final_loss": history[-1],
            }
        )
    return reports
