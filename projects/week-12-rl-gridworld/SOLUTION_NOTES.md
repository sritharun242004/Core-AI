# Solution notes — optimization and reinforcement learning

## 1. The MDP contract

A finite Markov decision process is `(S, A, P, R, γ)`: states, actions,
transition distribution, reward, and a discount factor. In this project the
transition distribution is deterministic, so `P(s' | s, a)` is one for the
single successor returned by `Gridworld.step` and zero elsewhere. The Markov
property is visible in the transition table: the next state and reward depend
on the current cell and action, not on the full path used to arrive there.

The episode begins at `(0, 0)` and ends at the goal or a time limit. Boundary
and wall moves are self-loops with the step penalty. The goal row is absorbing
in the model table. `step()` refuses to continue after termination, which
prevents accidentally collecting the goal reward twice.

A reward curve must be labeled carefully. This fixture is a known finite MDP,
so a greedy evaluation after learning is useful as a plumbing check. It is not
a held-out environment distribution and says nothing about transfer to a new
map.

## 2. Bellman optimality and tabular Q-learning

For a policy-independent action value, the optimality equation is:

```text
Q*(s, a) = E[r + γ (1 − done) max_a' Q*(s', a') | s, a]
```

A sample-based update moves one table entry toward its target:

```text
target = r                                      if terminal
         = r + γ max_a' Q(s', a')              otherwise
Q(s,a) ← Q(s,a) + α [target − Q(s,a)]
```

For `Q=2`, `r=1`, `max Q'=5`, `α=.2`, and `γ=.9`, the target is `5.5` and
the new value is `2.7`. A terminal transition uses `1` rather than `5.5`; this
is the most important off-by-one boundary in a tiny implementation.

The learner uses epsilon-greedy action selection. Epsilon is initially high so
all actions are tried, then decays to a nonzero floor. Exploration and learning
are separate: changing epsilon changes which transitions are observed, while
alpha controls the table update. The seeded NumPy generator makes a run
reproducible without relying on global random state.

### Q-learning debugging checklist

- Print one transition tuple before inspecting the curve.
- Confirm that state ids are in `[0, n_states)` and action ids in
  `[0, n_actions)`.
- Bootstrap through a time-limit truncation but not through a true terminal
  goal; the `info["terminated"]` flag carries that distinction.
- Compare `done=True` and `done=False` on the same hand-written update.
- Evaluate with greedy actions only after training; evaluating epsilon-greedy
  behavior conflates exploration with policy quality.

## 3. Policy gradients and REINFORCE

A stochastic policy parameterizes a distribution over actions:

```text
πθ(a | s) = softmax(zθ(s))_a
```

A trajectory `τ` has return `G_t = Σ_{k≥0} γ^k r_(t+k)`. The policy-gradient
objective is `J(θ)=E[Σ_t G_t log πθ(a_t|s_t)]`, so a minimization loss is:

```text
L_REINFORCE = − mean_t [log πθ(a_t | s_t) G_t]
```

The implementation samples from `Categorical(probs=...)`, stores the selected
log-probabilities, computes backwards discounted returns, and backpropagates
that scalar loss. Return standardization is a variance-reduction convention,
not a change to the action distribution. The optional entropy term encourages
exploration early in a run:

```text
L = L_REINFORCE − β mean_s H(πθ(. | s))
```

The baseline is deliberately small: one-hot state features and an optional
`tanh` hidden layer. One-hot features preserve the tabular state meaning while
still allowing the optimizer to change logits through gradients. The training
loop clips gradient norm as a safety guard and records episode rewards,
losses, and a moving average.

### Probability and gradient checks

`softmax_probabilities` must satisfy non-negativity and row sums of one even
for large logits. Computing softmax through the framework implementation avoids
naive exponent overflow. The tests also backpropagate through a hand-written
log-probability vector, then verify finite gradients. A finite gradient does
not prove useful credit assignment; it only proves that this local objective is
connected.

## 4. SGD, Adam, AdamW, and regularization

For a parameter `θ` and gradient `g_t`, vanilla SGD is:

```text
θ_(t+1) = θ_t − η g_t
```

Adam keeps first and second moments:

```text
m_t = β1 m_(t−1) + (1−β1) g_t
v_t = β2 v_(t−1) + (1−β2) g_t²
m̂_t = m_t / (1−β1^t),  v̂_t = v_t / (1−β2^t)
θ_(t+1) = θ_t − η m̂_t / (sqrt(v̂_t) + ε)
```

AdamW separates decoupled weight decay from the adaptive gradient step:

```text
θ_(t+1) = (1 − η λ) θ_t − η m̂_t / (sqrt(v̂_t) + ε)
```

A coupled L2 penalty instead augments the objective:

```text
L_total = L_data + λ Σ_i θ_i²
∂L_total/∂θ_i = ∂L_data/∂θ_i + 2λθ_i
```

The helper exposes the coupled penalty and optimizer weight decay as separate
choices. Do not silently pass both unless the experiment intentionally applies
both. L1 uses `λ |θ|` and produces a sign subgradient away from zero; it is a
useful conceptual contrast but is not hidden in the reference trainer.

## 5. PPO-Clip: a labeled conceptual boundary

PPO reuses trajectories from an older policy. Given old and new log-probability
values, the probability ratio is:

```text
r_t(θ) = πθ(a_t|s_t) / πold(a_t|s_t)
       = exp(log πθ(a_t|s_t) − log πold(a_t|s_t))
```

The clipped surrogate is:

```text
L_CLIP = E_t [ min(r_t A_t,
                  clip(r_t, 1−ε, 1+ε) A_t) ]
```

For a positive advantage, increasing the ratio above `1+ε` no longer improves
the objective. For a negative advantage, reducing it below `1−ε` no longer
improves the objective. This asymmetry is why `min` must be applied after
multiplying by the signed advantage; clipping the ratio in isolation is not
the objective.

`ppo_clip_demo` makes this algebra executable. It intentionally does not claim
to train PPO: a complete implementation also needs an on-policy rollout
collector, value function, advantage estimator, minibatch epochs, policy/value
loss balancing, and monitoring for instability. Keeping the clip derivation
small makes the boundary testable without smuggling in an unverified trainer.

## What the tests establish

The reference suite has 23 tests covering:

- deterministic transitions, absorbing goal dynamics, walls, boundaries, and
  separate terminal/time-limit flags;
- exact Bellman targets and tabular updates, seeded repeatability, and improved
  greedy reward;
- discounted returns, normalized action probabilities, finite policy gradients,
  policy reward/success improvement, and policy output shapes;
- PPO ratio clipping/objective terms plus explicit SGD, Adam, AdamW, and L2
  choices; and
- the no-download/no-accelerator source contract and import side effects.

These tests are contracts for a teaching implementation. They are not evidence
of a general RL result or a complete modern policy-optimization system.
