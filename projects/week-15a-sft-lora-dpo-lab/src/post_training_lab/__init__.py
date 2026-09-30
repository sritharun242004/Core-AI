"""Offline Week 15a lab: supervised fine-tuning, LoRA, and preference objectives."""

from .data import PreferencePair, SFTExample, make_preference_pairs, make_sft_examples
from .lora import LoRALinear, count_parameters, inject_lora, mark_only_lora_trainable
from .model import TinyCausalLM, TinyPolicy
from .objectives import (
    dpo_batch_loss,
    dpo_loss,
    ipo_loss,
    kto_loss,
    orpo_loss,
    sequence_logprob,
    sft_loss,
    simpo_loss,
)
from .quantization import QuantizedTensor, dequantize_int8, qlora_recipe, quantize_int8
from .train import load_checkpoint, save_checkpoint, train_sft

__all__ = [
    "LoRALinear",
    "PreferencePair",
    "QuantizedTensor",
    "SFTExample",
    "TinyCausalLM",
    "TinyPolicy",
    "count_parameters",
    "dequantize_int8",
    "dpo_batch_loss",
    "dpo_loss",
    "inject_lora",
    "ipo_loss",
    "kto_loss",
    "load_checkpoint",
    "make_preference_pairs",
    "make_sft_examples",
    "mark_only_lora_trainable",
    "orpo_loss",
    "qlora_recipe",
    "quantize_int8",
    "save_checkpoint",
    "sequence_logprob",
    "sft_loss",
    "simpo_loss",
    "train_sft",
]
