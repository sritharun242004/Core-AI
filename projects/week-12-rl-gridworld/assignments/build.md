# Build (2–3 hours)

Implement and inspect a complete local experiment:

1. Use the deterministic `Gridworld` and print `render()` before and after a
   greedy episode.
2. Train tabular Q-learning for at least 200 episodes. Plot episode reward and
   epsilon, then evaluate the greedy table separately. Include the Bellman
   target and terminal flag in your notes.
3. Train the one-hot REINFORCE baseline with a fixed seed. Compare SGD, Adam,
   and AdamW at a matched learning rate; report the moving-average reward,
   success rate, and action probabilities at the start state.
4. Add one regularization ablation: no penalty, coupled L2 in the loss, or
   AdamW decay. Explain which objective or update rule changed.
5. Reproduce the PPO-Clip conceptual table for several ratios and advantages.
   Label it as an objective demonstration, not a trained PPO result.

## Experiment record

Include map, reward values, max steps, seed, episode count, alpha/learning
rate, gamma, epsilon schedule, optimizer, regularization, and greedy evaluation
protocol. Keep the reference tests offline and do not replace the deterministic
fixture with a downloaded benchmark.
