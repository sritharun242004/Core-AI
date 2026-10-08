# %% [markdown]
# # Autodiff tour — micrograd
# Live walkthrough: build a graph, run forward, run backward, GD on XOR.

# %%
import random

from micrograd import Value, nn

a = Value(2.0)
b = Value(-3.0)
c = (a * b + a.relu()) * (b.exp() + Value(0.1))
c.backward()
print(f"c = {c.data:.4f}, dc/da = {a.grad:.4f}, dc/db = {b.grad:.4f}")

# %% [markdown]
# ## Train an MLP on XOR
# A single-neuron linear model cannot separate XOR — you need at least one hidden layer.

# %%
random.seed(1337)
model = nn.MLP(2, [4, 4, 1])
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]
for step in range(300):
    loss = Value(0.0)
    for x, y in zip(xs, ys, strict=True):
        pred = model([Value(x[0]), Value(x[1])])
        pred_val = pred[0]
        loss = loss + (pred_val - Value(y)) ** 2
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    if step % 50 == 0:
        print(f"step {step:3d} — loss {loss.data:.4f}")
