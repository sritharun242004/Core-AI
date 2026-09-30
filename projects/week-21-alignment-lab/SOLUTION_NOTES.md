# Solution notes — Week 21

## 1. DPO from a KL-regularized preference objective

A standard preference-based RLHF pipeline starts from supervised fine-tuning, fits a reward model to pairwise judgments, then optimizes the policy against that reward with a reference-policy KL penalty. PPO is one possible optimizer; it is not synonymous with RLHF. With reward `r`, reference `πref` and positive coefficient `β`, the idealized objective is

```text
max_π E_x E_(y~π(.|x))[r(x,y)] - β E_x KL(π(.|x) || πref(.|x)).
```

The optimum has `π*(y|x) ∝ πref(y|x) exp(r(x,y)/β)`. Rearranging gives `r = β log(π*/πref) + β log Z(x)`. Under a Bradley–Terry preference model, `P(chosen ≻ rejected|x) = sigmoid(r_chosen-r_rejected)`. The prompt-specific normalization cancels in the reward difference, giving DPO's reference-relative logistic loss. This is Stanford/Rafailov et al. (2023), not a claim that DPO needs online policy rollouts or a separate learned reward head.

For policy probabilities chosen `.6`, rejected `.3`, and reference `.4`, `.4`:

```text
margin = log(.6/.3) - log(.4/.4) = log(2) ≈ .693147
β = .5
loss = log(1 + exp(-.5 * log(2))) ≈ .534800
∂loss/∂margin = -β * sigmoid(-β * margin) ≈ -.207107
```

The loss decreases as the chosen/reference advantage over rejected/reference increases. Both chosen and rejected absolute likelihoods can fall in a general LM while their ratio improves; that is why the report also measures chosen probability. With identical policy and reference, the relative margin is zero, even if the policy already favors one response, so the loss is `log(2)`.

The KL-derived meaning of β does not make it a universal monotone knob for empirical KL in finite-data DPO. In the loss it scales the margin and gradient, while in the idealized reward objective it controls a reference penalty. Validate actual drift, task quality and safety rather than asserting that doubling β always doubles a particular observed quantity.

## 2. Why “no gradient” is not enough

`freeze_reference` makes a deep copy, freezes parameters, clears stale gradients and selects eval mode. `train_dpo` runs reference scoring under `no_grad` once and excludes it from the optimizer. Parameter storage cannot be shared, even through detached wrappers. A train-mode BatchNorm layer could still mutate buffers under `no_grad`; this toy has no running-statistics buffers, and the API additionally requires eval mode. Tests snapshot every state-dict entry and verify no reference parameter changes or acquires gradients.

The policy really trains, but memorizing eight context/action preference pairs is not evidence of alignment generalization. It is a numerical contract: correct loss, stable finite values, correct gradient signs, independent reference, and improved chosen probability. The one-token policy removes tokenizer, causal mask and padding complexity without misrepresenting a summed completion score. A real text extension must restore those complexities using Week 15a's masking tests.

## 3. Refusal is not the objective by itself

For four cases (two allowed, two prohibited), suppose the assistant correctly completes one allowed task, refuses the other, refuses one prohibited task and complies with the other. Then:

- Overall refusals: `2/4 = .5`.
- Required-refusal rate: `1/2 = .5`.
- Overrefusal rate: `1/2 = .5`.
- Allowed task success: `1/2 = .5`.
- Safe success: `2/4 = .5`, provided neither successful trajectory attempted a denied tool call.

An always-refusing assistant scores `1` on required refusal but `1` on overrefusal and `0` on allowed task success. No single refusal number supports a safety conclusion. Report denominator sizes and raw case IDs. Twenty synthetic cases are neither representative nor sufficient to estimate rare failure rates.

`MockTools` enforces the policy outside the assistant. A denied call is logged even if the assistant catches the exception and later refuses. Counts use **attempts**, not only successes, so retries cannot silently bypass the budget. Returning a read-only event tuple prevents ordinary callbacks from editing the displayed log, but a malicious Python callback can still access internals or execute arbitrary Python: this is not process isolation. A real service needs host-enforced capabilities, independently checked authorization, network/filesystem isolation, approvals for consequential effects, audit integrity and a stop path.

The parser baseline gets perfect toy scores because its grammar was deliberately designed for this fixture. Do not use its scores as a before/after DPO comparison: it is a different system. Real model refusal labels require independent human/rubric validation, not trusting the model to mark its own response safe.

## 4. CAI is a supervision procedure, not just a system prompt

In public Anthropic Constitutional AI (Bai et al., 2022), principles guide self-critique and response revision; revised answers support supervised training. AI comparisons then support a preference model used in RL from AI feedback. People still choose principles, evaluate coverage and govern deployment. Automating preference labels does not eliminate value judgments, correlated judge errors or reward hacking. The lesson explains this pipeline; no provider calls, model critiques or CAI training are performed by this lab.

## 5. Three representation questions

**Logit lens:** “If we decoded this intermediate residual state using the final decoder, what distribution appears?” Always include the model's actual final normalization, not an arbitrary vector unit norm. The final-state identity test is a high-signal correctness check. A high logit is not a decoded thought, causal pathway or faithful chain of reasoning.

**Probe:** “Is a label linearly recoverable from these activations on independent data?” Fit means/scales on training data, detach activations, fit a linear logistic classifier, and freeze it before evaluation. The fixture's target is explicitly `x[0] > 0`, so high accuracy only confirms recovery of a planted signal. With real activations, split by source, conversation or subject to avoid template leakage; add shuffled-label and nuisance-feature controls. A probe can exploit information that the original model never uses.

**SAE:** “Can a sparse learned dictionary reconstruct the activation distribution?” For `z = ReLU(Wenc(x-center)+benc)` and `xhat = z D + center`, optimize

```text
MSE = sum_ij((xhat_ij-x_ij)^2) / (N * input_dimensions)
L1  = sum_ik(abs(z_ik)) / N
loss = MSE + λ * L1
```

The decoder directions in `D` have unit norm. Otherwise the model can multiply decoder norms and divide codes, reducing L1 without learning a sparser explanation. The coefficient depends on the exact averaging convention; changing MSE from an elementwise mean to a per-example sum changes the effective sparsity pressure.

For codes `[[0,2,0],[1,0,0]]`, mean L0 is `1`, mean L1 is `1.5`, firing rates are `[.5,.5,0]`, and dead fraction is `1/3`. Evaluate activity and reconstruction on the same named split. A dead feature on 96 test points may fire elsewhere. More L1 can kill features or lose useful information. Normalized directions, low MSE and sparsity do not guarantee unique or monosemantic features.

Our SAE is trained on synthetic activations, not on Claude or even the tiny DPO model. Public Anthropic dictionary-learning/Circuits work motivates the exercise but includes substantially different model scales, datasets and analyses. Causal claims would require controlled interventions, downstream behavior measurements and confound checks; none are implemented here.

## 6. Interview worked solution

**Prompt:** “Preference accuracy rose, safety refusals rose, and an SAE feature correlates with bad outputs. Would you ship?”

1. Recover model/reference/data/rubric versions and train/validation/test separation. Check chosen likelihood, relative margin and policy drift, not just preference accuracy.
2. Separate refusals on prohibited tasks from false refusals on allowed tasks. Inspect trajectories, denied tool attempts and task success by slice; preserve paired case IDs from Week 20.
3. Reject the leap from a correlated feature to a causal mechanism. Require held-out probe evaluation, SAE reconstruction/activity trade-offs and interventions with controls before a causal claim.
4. Gate actual capabilities outside the model. Stage a bounded rollout only if frozen acceptance criteria pass; specify incident ownership, rollback and private-data retention limits.

**Baseline signal:** correct DPO equation and metric denominators. **Senior signal:** no reference mutation, test isolation, false refusals and unsafe-attempt accounting. **Staff signal:** explicitly separates optimization, behavior and mechanistic evidence; makes operational, governance and uncertainty limits part of the decision.

## What surprised me / common traps

- Loss `.693` does not mean the initial policy assigns chosen probability `.5`; it means policy and reference have the same pairwise log-odds.
- The toy DPO margin becomes very large on separable training pairs. That is overfit-able classification, not a calibrated human utility scale.
- Perfect parser fixture scores coexist with zero evidence of semantic robustness.
- Four of twelve SAE features can be inactive on the heldout split while reconstruction is good. Reporting only MSE hides that fact.
- Sparsity, interpretability and causality are different properties. A compelling visualization cannot substitute for a controlled experiment.
