import numpy as np
import pytest
import torch
from learning_to_rank_timeseries import (
    TinyNBeats,
    forecast_fixture,
    rolling_forecast,
    rolling_origins,
    seasonal_naive,
    train_forecaster,
    training_windows,
)

torch.set_num_threads(1)


def test_training_windows_never_cross_prefix_end():
    x, y = training_windows(np.arange(10.0), lookback=4, horizon=2)
    assert x.shape == (5, 4) and y.shape == (5, 2)
    assert x[-1].tolist() == [4, 5, 6, 7]
    assert y[-1].tolist() == [8, 9]
    assert np.all(y[:, 0] == x[:, -1] + 1)


def test_rolling_origin_indices_and_seasonal_naive_long_horizon():
    assert rolling_origins(30, initial=18, horizon=4, step=4) == [18, 22, 26]
    assert seasonal_naive(np.array([1, 2, 3, 4, 5, 6]), horizon=5, season=3).tolist() == [
        4,
        5,
        6,
        4,
        5,
    ]
    with pytest.raises(ValueError):
        seasonal_naive(np.arange(2), horizon=3, season=3)


def test_actual_residual_blocks_reconstruct_residual_and_sum_forecasts():
    model = TinyNBeats(lookback=8, horizon=3, blocks=2)
    x = torch.randn(4, 8)
    result = model(x)
    residual, forecast = x, torch.zeros(4, 3)
    for block in model.blocks:
        backcast, contribution = block(residual)
        residual = residual - backcast
        forecast = forecast + contribution
    assert torch.allclose(result, forecast)
    result.square().mean().backward()
    assert model.blocks[0].backcast.weight.grad.abs().sum() > 0
    assert all(b.forecast.weight.grad.abs().sum() > 0 for b in model.blocks)
    # The final backcast has no downstream block and is intentionally unused.
    assert model.blocks[-1].backcast.weight.grad is None


def test_nbeats_training_is_real_and_reduces_loss():
    values = forecast_fixture(80)
    model, history, mean, scale = train_forecaster(values[:60], lookback=12, horizon=4, steps=100)
    assert history[-1] < history[0] * 0.1
    assert mean == pytest.approx(values[:60].mean())
    assert scale == pytest.approx(values[:60].std())
    assert model(
        torch.tensor(((values[48:60] - mean) / scale)[None], dtype=torch.float32)
    ).shape == (1, 4)


def test_rolling_forecast_has_no_future_data_or_preprocessing_leakage():
    series = forecast_fixture(70)
    changed = series.copy()
    changed[50:] += 10000
    kwargs = dict(initial=50, horizon=4, step=20, lookback=12, season=6, steps=50)
    original_report = rolling_forecast(series, **kwargs)
    changed_report = rolling_forecast(changed, **kwargs)
    a, b = original_report[0], changed_report[0]
    assert a["origin"] == 50
    assert np.array_equal(a["forecast"], b["forecast"])
    assert np.array_equal(a["baseline"], b["baseline"])
    assert a["mean"] == b["mean"] and a["scale"] == b["scale"]
    assert a["mae"] != b["mae"]  # Future values are used only as scored targets.


def test_constant_series_stays_finite_and_invalid_geometry_rejected():
    _, _, mean, scale = train_forecaster(np.ones(30), lookback=6, horizon=2, steps=2)
    assert mean == 1 and scale == 1
    with pytest.raises(ValueError):
        training_windows(np.arange(5), lookback=4, horizon=3)
    with pytest.raises(ValueError):
        rolling_origins(10, initial=9, horizon=2, step=1)
    with pytest.raises(ValueError):
        training_windows(np.array([0, float("nan"), 1]), lookback=1, horizon=1)
