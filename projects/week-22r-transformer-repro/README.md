# Track R alpha — Transformer reproduction

Reproduce a mechanism, then design a credible experiment. This local PyTorch lab implements scaled causal attention and a one-block decoder with explicit Q/K/V, residuals, normalization, positions and FFN. It tests analytic attention, finite-difference gradients, causal invariance, next-token shifts, deterministic optimization and state-dict round trips.

```bash
cd projects/week-22r-transformer-repro
uv run --extra dev pytest
uv run python notebooks/01-transformer-reproduction.py
```

Public API: `attention(q,k,v)` returns output and weights; `TinyTransformer` maps `[B,T]` IDs to `[B,T,V]` logits; `causal_batch` builds shifted windows; `train_fixture` returns a seeded loss curve. No dataset/model downloads. Set `OMP_NUM_THREADS=1` for fast small CPU tests.

## Reproduction boundary

Vaswani et al. 2017 used an encoder-decoder translation model. This reference is decoder-only, pre-norm, GELU, learned positions, one layer and a periodic synthetic token fixture. Those are explicit deviations, not paper-equivalent results. Consult Week 13 for a broader decoder/SSM comparison. Full translation reproduction would require licensed data, tokenization, optimizer schedule, batching, validation BLEU protocol and a separate compute budget. The build assignment asks for an experiment manifest and one controlled ablation, not a fabricated original score.

Use the paper-reading protocol: extract the claim, identify the implementation unit, derive it independently, write an oracle, reproduce a curve, then report where the experiment departs from the source.
