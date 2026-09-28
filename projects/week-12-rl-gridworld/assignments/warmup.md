# Warmup (30–45 minutes)

## Deliverable

1. Draw the 4×4 state map and write the integer encoding for `(0, 0)`, `(3, 0)`,
   and `(3, 3)`. List the four action ids in order.
2. Run `Gridworld().transition_table()` and inspect a boundary self-loop, one
   move toward the goal, and the absorbing goal row.
3. Work one Bellman update by hand with `Q=2`, `r=1`, `max Q'=5`, `α=.2`, and
   `γ=.9`, once for a nonterminal transition and once for a terminal one.
4. Write the REINFORCE loss for a three-action policy and prove that a softmax
   row sums to one.
5. Run the PPO clip demo with ratios `0.5`, `1.0`, and `1.5`, using both a
   positive and a negative advantage. Explain why the signed `min` matters.

## Check

Start by writing a failing test for one boundary transition or Bellman target,
then make the smallest implementation change that passes it. Record the seed
and the exact command used; do not infer policy quality from one sampled
trajectory.
