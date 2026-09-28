# Warmup — read the skip path (30–45 min)

1. Instantiate `BasicBlock(8, 8)` and `BasicBlock(8, 16, stride=2)`.
2. Draw the tensor shapes after each convolution and after each identity or
   projection branch.
3. Run the block on a `(2, 8, 16, 16)` tensor and explain why the second block
   returns `(2, 16, 8, 8)`.
4. Compute a mean cross-entropy loss on the output of a tiny model, call
   `backward()`, and print the first convolution's gradient norm.

**Deliverable:** a short shape table and one sentence explaining why an
identity skip cannot be used when channels or resolution change. Start by
writing a failing shape assertion, then implement or repair the assertion.
