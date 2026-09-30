"""Expanding-window rolling origins and actual tiny residual backcast blocks."""

import numpy as np
import torch
from torch import nn


def _series(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("series must be a nonempty finite one-dimensional array")
    return result


def forecast_fixture(length: int = 100):
    if length < 1:
        raise ValueError("length must be positive")
    t = np.arange(length)
    return 10 + 0.04 * t + 2 * np.sin(2 * np.pi * t / 6) + 0.2 * np.cos(2 * np.pi * t / 3)


def training_windows(prefix, *, lookback: int, horizon: int):
    values = _series(prefix)
    if min(lookback, horizon) < 1 or len(values) < lookback + horizon:
        raise ValueError("prefix must fit at least one complete input/target window")
    starts = range(lookback, len(values) - horizon + 1)
    return np.stack([values[t - lookback : t] for t in starts]), np.stack(
        [values[t : t + horizon] for t in starts]
    )


def rolling_origins(length: int, *, initial: int, horizon: int, step: int):
    if min(initial, horizon, step) < 1 or initial + horizon > length:
        raise ValueError("positive geometry and at least one complete forecast required")
    return list(range(initial, length - horizon + 1, step))


def seasonal_naive(history, *, horizon: int, season: int):
    values = _series(history)
    if min(horizon, season) < 1 or len(values) < season:
        raise ValueError("need a complete past season and a positive horizon")
    return np.resize(values[-season:], horizon).copy()


class ResidualBlock(nn.Module):
    """Generic learned backcast/forecast heads; no fixed trend/seasonal basis."""

    def __init__(self, lookback: int, horizon: int, width: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(lookback, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.backcast = nn.Linear(width, lookback)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, residual):
        hidden = self.hidden(residual)
        return self.backcast(hidden), self.forecast(hidden)


class TinyNBeats(nn.Module):
    """N-BEATS-like generic residual stack, not the full published architecture.

    x_next = x - backcast; y_hat = sum(forecasts). The last backcast is unused
    by forecast loss because there is no downstream residual block.
    """

    def __init__(
        self, lookback: int, horizon: int, *, width: int = 32, blocks: int = 2, seed: int = 23
    ):
        super().__init__()
        if min(lookback, horizon, width, blocks) < 1:
            raise ValueError("positive architecture dimensions required")
        self.lookback, self.horizon = lookback, horizon
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.blocks = nn.ModuleList(
                [ResidualBlock(lookback, horizon, width) for _ in range(blocks)]
            )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.lookback:
            raise ValueError("expected [batch, lookback] inputs")
        residual = x
        prediction = x.new_zeros((len(x), self.horizon))
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            prediction = prediction + forecast
        return prediction


def train_forecaster(
    prefix,
    *,
    lookback: int = 12,
    horizon: int = 4,
    steps: int = 100,
    lr: float = 0.01,
    seed: int = 23,
):
    """Fit preprocessing AND weights on prefix only; no validation/test tuning."""
    values = _series(prefix)
    if steps < 1 or lr <= 0:
        raise ValueError("positive steps and learning rate required")
    mean = float(values.mean())
    scale = float(values.std())
    if scale < 1e-8:
        scale = 1.0
    x, y = training_windows((values - mean) / scale, lookback=lookback, horizon=horizon)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    model = TinyNBeats(lookback, horizon, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps + 1):
        loss = (model(x) - y).square().mean()
        history.append(float(loss.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, history, mean, scale


def rolling_forecast(
    series,
    *,
    initial: int = 60,
    horizon: int = 4,
    step: int = 4,
    lookback: int = 12,
    season: int = 6,
    steps: int = 100,
    seed: int = 23,
):
    """Refit independently at each origin; targets start at origin, not origin+1.

    Returns raw forecast/target arrays and errors, not an invalid iid window CI.
    Overlapping horizons create dependent errors even with a correct time split.
    """
    values = _series(series)
    reports = []
    for origin in rolling_origins(len(values), initial=initial, horizon=horizon, step=step):
        prefix = values[:origin]
        model, history, mean, scale = train_forecaster(
            prefix, lookback=lookback, horizon=horizon, steps=steps, seed=seed
        )
        inputs = torch.tensor(((prefix[-lookback:] - mean) / scale)[None], dtype=torch.float32)
        with torch.no_grad():
            forecast = model(inputs)[0].numpy() * scale + mean
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
