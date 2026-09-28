# Challenge (3+ hrs)

Extend `engine.py` to support **broadcasting** on a `Tensor` class that holds a numpy array as its `.data` and its `.grad`. Rules:

1. Same public API (`+`, `*`, `**`, `.relu()`, `.backward()`).
2. `Tensor.dot(Tensor)` must work — this is the primitive that lets you replace the per-scalar `Neuron` with a per-Layer matrix multiply.
3. Retrain the moons MLP with your `Tensor` class. Measure: is it faster than the scalar version? How much? Post the numbers in `SOLUTION_NOTES.md`.
4. Bonus: get the numeric-parity test to pass against `torch` on a Tensor-graph. (`atol=1e-5` is fine.)

Stretch: implement `.softmax()` and reproduce the 3-class Iris classifier from `torch.nn.CrossEntropyLoss`.
