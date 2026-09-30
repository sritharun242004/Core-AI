"""Week 13: an inspectable causal transformer beside an explicit SSM baseline."""

from .data import TINY_TEXT, CharVocab, TinyCharDataset, make_tiny_dataset
from .model import (
    MLP,
    CausalSelfAttention,
    GPTConfig,
    GPTLanguageModel,
    NanoGPT,
    NanoGPTLM,
    PositionalEncoding,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    TransformerLanguageModel,
)
from .ssm import (
    ExplicitSSMCell,
    MambaStyleBaseline,
    SelectiveSSM,
    TinySSM,
    TinySSMLanguageModel,
    diagonal_ssm_scan,
    ssm_recurrence,
)
from .train import fit, language_model_loss, train_epoch, training_step

__all__ = [
    "MLP",
    "TINY_TEXT",
    "CausalSelfAttention",
    "CharVocab",
    "ExplicitSSMCell",
    "GPTConfig",
    "GPTLanguageModel",
    "MambaStyleBaseline",
    "NanoGPT",
    "NanoGPTLM",
    "PositionalEncoding",
    "SelectiveSSM",
    "SinusoidalPositionalEncoding",
    "TinyCharDataset",
    "TinySSM",
    "TinySSMLanguageModel",
    "TransformerBlock",
    "TransformerLanguageModel",
    "diagonal_ssm_scan",
    "fit",
    "language_model_loss",
    "make_tiny_dataset",
    "ssm_recurrence",
    "train_epoch",
    "training_step",
]
