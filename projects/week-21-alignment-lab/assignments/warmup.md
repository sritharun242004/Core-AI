# Warmup — three claims, three denominators (30 minutes)

No training or network needed. Submit a short calculation sheet before opening the solution notes.

1. Policy probabilities for chosen/rejected responses are `.6/.3`; the frozen reference uses `.4/.4`. Compute the reference-relative DPO margin, loss at β `.5`, and the sign of each policy-score gradient. Repeat when policy equals reference. Explain why reference log-probabilities must not receive gradients.
2. A four-case battery contains two allowed and two prohibited tasks. One allowed task succeeds, the other is refused; one prohibited task is refused, the other is not. Compute overall refusal, required refusal, overrefusal, task success and safe success. State every denominator. Recompute safe success if the correct refusal follows a blocked tool attempt.
3. For SAE codes `[[0,2,0],[1,0,0]]`, compute mean L0, mean L1, per-feature firing rates and dead fraction. State the threshold used for activity.
4. Explain why a 98% held-out probe accuracy and a sparse feature correlated with refusal do not establish a causal refusal circuit.

**Acceptance:** DPO loss about `.5348`; five clearly distinguished safety metrics; activity statistics with explicit averaging conventions; no claim that a probe proves causal use. Use `SOLUTION_NOTES.md` only after attempting the derivations.
