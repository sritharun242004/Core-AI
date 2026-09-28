# Solution notes — micrograd

- Every op returns a fresh `Value`; you compose forward from left to right.
- `_backward` is a closure — it captures the parent Values and each op's local gradient rule.
- `backward()` topo-sorts the graph, seeds `self.grad = 1.0`, then walks reverse-topo calling every `_backward()`.
- Common pitfall: forgetting to zero grads between GD steps → gradients accumulate (intentional in PyTorch too; that's why `optimizer.zero_grad()` exists).
- The XOR example needs at least one hidden layer of ≥ 2 neurons — a single-neuron linear model cannot separate XOR.

## Gotchas found while writing this

- `__radd__` and `__rmul__` are needed so `Value(2) + 3` and `3 + Value(2)` both work.
- `Value` must be trackable-by-id (using `__slots__` gives us hashable-by-id for free) so it can be stored in a `set` for topo visit tracking.
- `log(x)` must guard `x > 0` — silent NaN propagation was the original micrograd's biggest gotcha.
- At a branching `Value` (used in two downstream ops), gradients from each downstream path **add**. This is the multivariable chain rule; forgetting `+=` (writing `=` instead) is the #1 micrograd bug.
