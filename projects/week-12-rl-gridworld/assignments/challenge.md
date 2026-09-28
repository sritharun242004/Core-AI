# Challenge (3+ hours)

Choose one path and pre-register the measurement before running it.

## A. Generalization across maps

Create a family of small maps with fixed wall-generation seeds. Train on one
map and evaluate on unseen maps with the same state/action contract. Compare a
Q-table with the policy network, report success and return distributions, and
explain why one-hot state ids make transfer difficult.

## B. Advantage and baseline study

Add a learned value baseline or a leave-one-out return baseline to REINFORCE.
Compare return variance, gradient norms, and success rate at matched seeds and
updates. Check that the baseline is not allowed to use future evaluation data.

## C. Minimal PPO extension

Add a rollout buffer, old log-probabilities, discounted advantages, and a value
head. Reuse `ppo_clip_demo` for the objective and add tests for ratio clipping,
shape contracts, and finite gradients. Clearly separate the new trainer from
the reference REINFORCE baseline and report clip epsilon, epochs, minibatch
size, and any KL guard.

## Challenge rubric

- **Baseline:** tests dynamics and reports reward with a fixed evaluation policy.
- **Senior:** separates exploration, optimization, variance reduction, and map
  distribution; includes at least three seeds.
- **Staff:** pre-registers metrics, failure cases, compute budget, and evidence
  boundaries instead of turning a toy return into a production claim.
