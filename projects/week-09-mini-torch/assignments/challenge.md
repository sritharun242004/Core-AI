# Challenge — extend the engine without hiding the derivative (3+ hrs)

Implement one extension and defend it with a numerical test:

- momentum SGD or Adam, including a short comparison against plain SGD;
- a `sigmoid` or `softmax` primitive with a stable forward pass;
- a finite-difference gradient checker for a random MLP parameter; or
- a tiny convolution-free residual block composed from existing operations.

Your submission should contain:

1. a failing test written first;
2. the implementation with an explicit local backward rule;
3. a comparison against PyTorch or central finite differences;
4. a paragraph naming the edge cases your implementation does not support.

**Stretch:** implement gradient clipping and show how it changes an intentionally
large update. Do not silently switch to PyTorch for the update itself.
