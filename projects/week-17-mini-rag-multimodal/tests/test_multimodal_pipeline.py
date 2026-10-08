# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false
import socket

import numpy as np
import pytest
from mini_rag_multimodal import DOCUMENTS, Retriever, VoicePipeline
from mini_rag_multimodal.generation import denoise_step, tiny_vae_roundtrip
from mini_rag_multimodal.pipeline import answer_question


def test_offline_end_to_end_extracts_evidence_with_citations(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    answer = answer_question(Retriever(DOCUMENTS), "diffusion noise", k=2)
    assert answer.citations[0] == "diffusion"
    assert "[diffusion]" in answer.text
    assert "denoising" in answer.text
    voice = VoicePipeline(Retriever(DOCUMENTS)).run(b"diffusion noise", k=2)
    assert voice.transcript == "diffusion noise"
    assert "[diffusion]" in voice.response
    assert voice.audio.startswith(b"OFFLINE_AUDIO:")
    assert answer_question(Retriever(DOCUMENTS), "zzzzzz").citations == ()
    assert "no local evidence" in VoicePipeline(Retriever(DOCUMENTS)).run("").response


def test_voice_interfaces_accept_injected_adapters() -> None:
    class FakeASR:
        def transcribe(self, audio):
            assert audio == b"fixture"
            return "speech"

    class FakeTTS:
        def synthesize(self, text):
            assert "[voice]" in text
            return b"not-real-audio"

    result = VoicePipeline(Retriever(DOCUMENTS), FakeASR(), FakeTTS()).run(b"fixture")
    assert result.audio == b"not-real-audio"


def test_diffusion_training_step_reduces_fixed_batch_noise_prediction_loss() -> None:
    image = np.full((8, 8), 0.5)
    result = denoise_step(image, noise_scale=0.2, seed=7)
    again = denoise_step(image, noise_scale=0.2, seed=7)
    assert np.isfinite(result.loss_before) and np.isfinite(result.loss_after)
    assert result.loss_after < result.loss_before
    assert np.array_equal(result.restored, again.restored)
    assert np.isfinite(result.gradient_norm) and result.gradient_norm > 0
    # The forward process must be un-clipped Gaussian corruption.
    assert np.array_equal(result.noisy, again.noisy)


def test_vae_exposes_sampled_posterior_and_finite_elbo_terms() -> None:
    image = np.full((4, 4), 0.5)
    result = tiny_vae_roundtrip(image, latent_size=2, seed=5)
    assert result.mean.shape == result.log_variance.shape == result.latent.shape
    assert not np.array_equal(result.latent, result.mean)
    assert result.kl >= 0 and np.isfinite(result.loss)
    assert result.loss == pytest.approx(result.reconstruction_loss + result.kl)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.2])
def test_denoise_rejects_invalid_scale(value: float) -> None:
    with pytest.raises(ValueError, match="noise_scale"):
        denoise_step(np.zeros((2, 2)), noise_scale=value)


def test_generate_rejects_nonfinite_and_nonimage_input() -> None:
    for image in (np.array([1.0]), np.array([[np.nan]]), np.zeros((0, 2))):
        with pytest.raises(ValueError):
            tiny_vae_roundtrip(image)
        with pytest.raises(ValueError):
            denoise_step(image)
