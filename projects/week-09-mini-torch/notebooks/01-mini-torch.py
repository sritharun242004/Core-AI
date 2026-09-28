# %% [markdown]
# Week 9 — mini-torch from first principles
#
# This notebook is percent-format Python. It uses a deterministic synthetic
# 28x28 seven-segment fixture so it runs offline; it is not a real-MNIST claim.

# %%
from mini_torch import MLP, Tensor, accuracy, cross_entropy, make_mnist_shaped, train_mlp

# %% [markdown]
# ## 1. A scalar loss is a graph
#
# `Tensor.backward()` walks parents in reverse topological order. The same
# machinery works for a matrix multiplication and a batch reduction.

# %%
x = Tensor([[0.2, -0.4], [1.1, 0.7]], requires_grad=True)
w = Tensor([[0.3, -0.8, 0.2], [0.5, 0.6, -0.1]], requires_grad=True)
b = Tensor([0.1, -0.2, 0.05], requires_grad=True)
loss = ((x @ w + b).relu() ** 2).mean()
loss.backward()
print("loss:", loss.item())
print("d loss / d w:\n", w.grad)

# %% [markdown]
# ## 2. Train an MLP

# %%
x_train, y_train, x_test, y_test = make_mnist_shaped(
    n_train=500, n_test=100, noise=0.04, seed=7
)
model = MLP(28 * 28, hidden=(64, 32), out_features=10, seed=7)
history = train_mlp(
    model,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1,
    batch_size=50,
    seed=7,
)
print("final train loss:", history["loss"][-1])
print("train accuracy:", accuracy(model(x_train), y_train))
print("test accuracy:", accuracy(model(x_test), y_test))

# %% [markdown]
# ## 3. Inspect reduction semantics
#
# `none` exposes one loss per row. `sum` and `mean` differ only by the scale
# factor, but that factor directly affects an SGD update.

# %%
logits = model(x_test[:8])
for reduction in ("none", "sum", "mean"):
    value = cross_entropy(logits, y_test[:8], reduction=reduction)
    print(reduction, value.data)
