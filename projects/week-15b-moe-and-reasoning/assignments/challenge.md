# Challenge — make the caveats measurable

Choose one extension and add tests first:

- Replace deterministic first-come capacity with a documented token policy
  (for example, priority by router score), then compare dropped rates and
  finite gradients.
- Add noisy arithmetic responses and compare single-sample accuracy with
  majority vote over several group sizes. Report confidence intervals over
  several explicit seeds.
- Add a held-out arithmetic template split and test whether the verifier's
  syntax boundary is being mistaken for reasoning generalization.
- Add a dense feed-forward baseline with a matched parameter/update budget.
  Report task loss, auxiliary loss, expert utilization, wall-clock time, and
  dropped assignments rather than only the final loss.
- Add a reference-policy penalty to the generic `grpo_loss` and explain what
  is local design versus what the cited public GRPO/R1 paper establishes.

Do not claim that a tiny MLP is DeepSeek-MoE, that a toy relative-advantage
update reproduces DeepSeek-R1, or that majority vote proves reasoning. Every
comparison should include seed, device, token/update budget, architecture,
verifier, and failure cases.
