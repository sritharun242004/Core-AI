# Solutions and research notes

Zero queries/keys give uniform probability over permitted prefix positions. Values [1,3,8] therefore yield [1,2,4], not the full-sequence mean at every position. Mask before softmax; masking probabilities afterward without renormalizing changes the result. Compare Q/K/V gradients using double precision and finite differences. Half-precision Q/K products accumulate in float32 before scaling/softmax to avoid representable scaled scores overflowing in the unscaled product; float64 inputs retain float64 arithmetic. Reject nonfinite inputs and score overflow. CPU fixture helpers seed only the CPU generator and restore it, leaving accelerator RNG streams untouched.

Causal invariance changes suffix IDs and asserts unchanged prefix logits. A diagonal-only or transposed triangular mask fails a different oracle, so retain both analytic and perturbation tests. Inputs are each window except the last token; labels are that window except the first.

The decoder uses pre-norm for convenient stable small training, but the original Transformer used a different normalization layout. Learned positions and GELU are also deviations. List each in the writeup rather than silently calling the model identical. A checkpoint needs architecture and vocabulary metadata in a real experiment; this test holds constructor configuration fixed.

A reproduction table should include source claim, implementation file, oracle, dataset revision, seed, metric and deviation. An ablation changes one mechanism at a time and preserves evaluation. Report failed reproductions and negative results. Interview signals: baseline derives attention; senior checks gradients and token shifts; staff separates implementation parity, experimental parity and generalization.
