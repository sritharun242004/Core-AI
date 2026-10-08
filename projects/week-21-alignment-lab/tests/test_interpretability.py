import pytest
import torch
import torch.nn.functional as functional
from alignment_lab import (
    TinyPolicy,
    TinySAE,
    activation_metrics,
    fit_linear_probe,
    logit_lens,
    probe_accuracy,
    probe_fixture,
    sae_fixture,
    sae_loss,
    train_sae,
)
from alignment_lab.torch_api import backward


def test_logit_lens_uses_final_normalization_and_unembedding():
    model = TinyPolicy(seed=21)
    contexts = torch.arange(8)
    states = model.hidden_states(contexts)
    decoded = logit_lens(states, model.final_norm, model.unembedding)
    assert decoded.shape == (8, 2, 3)
    assert torch.allclose(decoded[:, -1], model(contexts))
    expected = model.unembedding(model.final_norm(states))
    assert torch.allclose(decoded, expected)
    assert not torch.allclose(decoded[:, 0], model.unembedding(states[:, 0]))
    assert torch.allclose(decoded.softmax(-1).sum(-1), torch.ones(8, 2))


def test_probe_is_fit_on_train_only_and_generalizes_on_independent_holdout():
    train_x, train_y, test_x, test_y = probe_fixture(seed=21)
    probe = fit_linear_probe(train_x, train_y, seed=21)
    assert torch.allclose(probe.mean, train_x.mean(0))
    assert torch.allclose(probe.scale, train_x.std(0, unbiased=False).clamp_min(1e-6))
    assert probe_accuracy(probe, test_x, test_y) > 0.9
    assert probe_accuracy(probe, test_x, 1 - test_y) < 0.1
    assert probe_accuracy(probe, train_x, train_y) > 0.95
    assert not torch.equal(train_x[: len(test_x)], test_x)
    again = fit_linear_probe(train_x, train_y, seed=21)
    assert all(
        torch.equal(a, b) for a, b in zip(probe.parameters(), again.parameters(), strict=True)
    )


def test_sae_objective_is_reconstruction_plus_l1_with_gradients():
    model = TinySAE(4, 6, seed=21)
    values = torch.tensor([[1.0, -1.0, 2.0, 0.5], [2.0, 0.0, -1.0, 1.0]])
    reconstruction, codes = model(values)
    loss = sae_loss(values, reconstruction, codes, l1_coefficient=0.07)
    expected = functional.mse_loss(reconstruction, values) + 0.07 * codes.abs().sum(-1).mean()
    assert torch.allclose(loss, expected)
    backward(loss)
    assert model.encoder.weight.grad is not None
    assert model.decoder.grad is not None
    assert (codes >= 0).all()
    assert torch.allclose(model.dictionary.square().sum(dim=1).sqrt(), torch.ones(6), atol=1e-6)


def test_sae_training_improves_heldout_reconstruction_and_reports_activations():
    train_x, test_x = sae_fixture(seed=21)
    result = train_sae(train_x, features=12, steps=250, seed=21)
    assert result.final_reconstruction < result.initial_reconstruction * 0.4
    assert result.losses[-1] < result.losses[0]
    reconstruction, codes = result.model(test_x)
    assert (
        functional.mse_loss(reconstruction, test_x) < ((test_x - train_x.mean(0)) ** 2).mean() * 0.4
    )
    metrics = activation_metrics(codes)
    assert 0 < metrics.mean_l0 < 12
    assert metrics.mean_l1 > 0
    assert 0 <= metrics.dead_fraction <= 1
    assert len(metrics.firing_rates) == 12
    assert all(0 <= frequency <= 1 for frequency in metrics.firing_rates)
    assert torch.allclose(result.model.center, train_x.mean(0))


def test_activation_metrics_have_explicit_denominators():
    metrics = activation_metrics(torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]))
    assert metrics.mean_l0 == 1
    assert metrics.mean_l1 == 1.5
    assert metrics.dead_fraction == pytest.approx(1 / 3)
    assert metrics.firing_rates == (0.5, 0.5, 0.0)


def test_invalid_interpretability_inputs_are_rejected():
    with pytest.raises(ValueError):
        activation_metrics(torch.empty(0, 3))
    with pytest.raises(ValueError):
        fit_linear_probe(torch.ones(4, 3), torch.tensor([0, 1, 2, 0]))
    with pytest.raises(ValueError):
        train_sae(torch.ones(4, 3), l1_coefficient=-1)
    with pytest.raises(ValueError):
        sae_loss(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 4), l1_coefficient=-1)
