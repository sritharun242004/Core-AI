# %% [markdown]
# Week 10 — CNNs and residual networks on CIFAR-shaped data
#
# This percent-format notebook is offline by default. It uses the deterministic
# fake fixture for shape and training plumbing; it does not claim CIFAR-10
# accuracy. Opt into the real loader only after reading COMPUTE.md.

# %%
import torch
from cifar_resnet import (
    CIFARCNN,
    CIFARTrainTransform,
    count_parameters,
    make_fake_cifar,
    resnet18,
)
from cifar_resnet.train import fit
from torch.utils.data import DataLoader

# %% [markdown]
# ## 1. A convolution changes local receptive fields
#
# A 3x3 convolution applies the same learned weights at every location. The
# baseline uses stride 2 after its first block and global average pooling
# before classification, so it does not hard-code a flatten size.

# %%
x = torch.randn(4, 3, 32, 32)
cnn = CIFARCNN(num_classes=10, channels=(8, 16, 32))
print("CNN parameters:", count_parameters(cnn))
print("CNN logits:", cnn(x).shape)

# %% [markdown]
# ## 2. The residual route
#
# A BasicBlock computes `ReLU(F(x) + S(x))`. `S` is identity when shapes match
# and a learned 1x1 projection when a stage changes channels or resolution.

# %%
resnet = resnet18(num_classes=10, widths=(8, 16, 32, 64), blocks=(1, 1, 1, 1))
logits = resnet(x)
print("ResNet parameters:", count_parameters(resnet))
print("ResNet logits:", logits.shape)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
loss.backward()
print("first gradient norm:", resnet.stem[0].weight.grad.norm().item())

# %% [markdown]
# ## 3. Offline smoke training
#
# The synthetic labels are paired with a weak, deterministic image signal so a
# short run can validate the loop. It is not a substitute for CIFAR-10.

# %%
train = make_fake_cifar(64, seed=7, transform=CIFARTrainTransform(padding=2))
history = fit(
    resnet18(num_classes=10, widths=(4, 8), blocks=(1, 1)),
    DataLoader(train, batch_size=16, shuffle=False),
    epochs=2,
    learning_rate=0.02,
    device="cpu",
)
print(history)

# %% [markdown]
# ## 4. Optional transfer-learning boundary
#
# `freeze_backbone` and `replace_classifier` intentionally stay small. A real
# pretrained backbone needs an explicit normalization policy and a checkpoint
# provenance note before its score is compared with a from-scratch run.
