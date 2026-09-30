"""Tiny NumPy generative demonstrations, not pretrained image models."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VAEResult:
    latent: np.ndarray
    reconstruction: np.ndarray
    mean: np.ndarray
    log_variance: np.ndarray
    reconstruction_loss: float
    kl: float
    loss: float


@dataclass(frozen=True)
class DenoiseResult:
    clean: np.ndarray
    noisy: np.ndarray
    restored: np.ndarray
    loss_before: float
    loss_after: float
    gradient_norm: float

    def __iter__(self):
        yield self.clean
        yield self.noisy
        yield self.restored


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("image must contain only finite values")
    return np.clip(array, 0.0, 1.0)


def tiny_vae_roundtrip(image: np.ndarray, latent_size: int = 4, seed: int = 0) -> VAEResult:
    """Encode/decode one tiny grayscale image with a reparameterized latent.

    This demonstrates the shape of an encoder/decoder and ELBO terms; it does not
    train, sample a useful posterior, or claim image quality.
    """
    if latent_size <= 0:
        raise ValueError("latent_size must be positive")
    array = _validate_image(image)
    rng = np.random.default_rng(seed)
    pixels = array.reshape(-1)
    encoder = rng.normal(0.0, 0.4, size=(latent_size, pixels.size))
    log_variance = np.full(latent_size, -1.0)
    mean = np.tanh(encoder @ pixels)
    latent = mean + np.exp(0.5 * log_variance) * rng.normal(size=latent_size)
    decoder = rng.normal(0.0, 0.4, size=(pixels.size, latent_size))
    reconstruction = 1 / (1 + np.exp(-(decoder @ latent)))
    reconstruction = reconstruction.reshape(array.shape)
    reconstruction_loss = float(np.mean((reconstruction - array) ** 2))
    kl = float(0.5 * np.sum(np.exp(log_variance) + mean**2 - 1 - log_variance))
    return VAEResult(
        latent=latent,
        reconstruction=reconstruction,
        mean=mean,
        log_variance=log_variance,
        reconstruction_loss=reconstruction_loss,
        kl=kl,
        loss=reconstruction_loss + kl,
    )


def denoise_step(image: np.ndarray, noise_scale: float = 0.2, seed: int = 0) -> DenoiseResult:
    """Add Gaussian noise and take one local mean-based denoising step.

    The local mean is a deliberately tiny stand-in for a learned diffusion noise
    predictor. One step is useful for tracing tensors, not for image generation.
    """
    if noise_scale < 0 or not np.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and nonnegative")
    clean = _validate_image(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=clean.shape)
    noisy = np.clip(clean + noise_scale * noise, 0.0, 1.0)
    padded = np.pad(noisy, 1, mode="edge")
    neighbours = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4
    restored = np.clip(0.5 * noisy + 0.5 * neighbours, 0.0, 1.0)
    # Compare image reconstruction error before and after one denoising update.
    # This is a compact analogue of an optimization-step contract, not training.
    target_noise = noise_scale * noise
    loss_before = float(np.mean((noisy - clean) ** 2))
    predicted_noise = restored - clean
    loss_after = float(np.mean((restored - clean) ** 2))
    gradient_norm = float(np.linalg.norm(target_noise - predicted_noise))
    return DenoiseResult(
        clean=clean.copy(),
        noisy=noisy,
        restored=restored,
        loss_before=loss_before,
        loss_after=loss_after,
        gradient_norm=gradient_norm,
    )
