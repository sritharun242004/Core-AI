"""Offline Week 17 mini-RAG and multimodal teaching lab."""

from .corpus import DOCUMENTS, QRELS, QUERIES, Document
from .generation import DenoiseResult, VAEResult, denoise_step, tiny_vae_roundtrip
from .metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from .pipeline import Answer, answer_question
from .retrieval import Hit, Retriever, reciprocal_rank_fusion, rerank
from .variants import graph_expand, hyde_query, late_interaction_search
from .voice import OfflineSynthesizer, OfflineTranscriber, VoicePipeline, VoiceResult

__all__ = [
    "DOCUMENTS",
    "QRELS",
    "QUERIES",
    "Answer",
    "DenoiseResult",
    "Document",
    "Hit",
    "OfflineSynthesizer",
    "OfflineTranscriber",
    "Retriever",
    "VAEResult",
    "VoicePipeline",
    "VoiceResult",
    "answer_question",
    "average_precision_at_k",
    "denoise_step",
    "evaluate_rankings",
    "graph_expand",
    "hyde_query",
    "late_interaction_search",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "reciprocal_rank_fusion",
    "rerank",
    "tiny_vae_roundtrip",
]
