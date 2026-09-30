# Build — train and compare two causal models (2–3 hrs)

1. Run the percent notebook and inspect the first attention probability rows.
2. Add a small held-out suffix or a second deterministic text fixture without
   downloading a corpus. Keep the vocabulary and split manifest explicit.
3. Train NanoGPT and TinySSM for the same number of tokens. Match seed,
   optimizer, learning rate, context length, and evaluation protocol first;
   then run a parameter-matched comparison.
4. Plot train and held-out token loss, count parameters, and report the mean and
   standard deviation across at least three seeds if you make a claim.
5. Explain what the causal-mask and SSM-recurrence tests establish before
   interpreting a loss curve.

**Deliverable:** a short experiment note with fixture provenance, model
configuration, token/update budget, seed(s), parameter counts, losses, device,
and one limitation. A lower loss on this tiny repeated text is not evidence of
better long-context language modeling.
