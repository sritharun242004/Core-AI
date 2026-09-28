# cifar-resnet — Week 10

A small, inspectable PyTorch lab for convolutional networks, residual
connections, augmentation, and transfer learning. The implementation is
intentionally dependency-light: PyTorch is required; torchvision is imported
only by the opt-in real-CIFAR loader. The tests use a deterministic,
CIFAR-shaped in-memory fixture and never download data.

## Run the offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
```

The notebook is percent-format Python and can be opened in Jupyter or converted
with Jupytext:

```bash
PYTHONPATH=src jupytext --to notebook notebooks/01-cifar-resnet.py
```

## Public API

```python
from torch.utils.data import DataLoader

from cifar_resnet import CIFARTrainTransform, make_fake_cifar, resnet18
from cifar_resnet.train import fit

train = make_fake_cifar(
    256,
    seed=7,
    transform=CIFARTrainTransform(),
)
model = resnet18(num_classes=10)
history = fit(
    model,
    DataLoader(train, batch_size=64, shuffle=True),
    epochs=2,
    learning_rate=0.1,
)
```

The core objects are:

- `CIFARCNN`: a compact convolutional baseline with global average pooling.
- `BasicBlock`, `ResNet`, and `resnet18`: CIFAR-friendly 3×3 residual blocks,
  projection skips when shape changes, and no ImageNet max-pool.
- `FakeCIFAR10` / `make_fake_cifar`: deterministic 3×32×32 float images with a
  deliberately learnable synthetic signal. This is a test fixture, not CIFAR.
- `CIFARTrainTransform`: tensor-only random crop and horizontal flip with an
  injectable `torch.Generator` for reproducible augmentation tests.
- `train_epoch`, `evaluate`, and `fit`: small device-aware loops with
  sample-weighted loss and accuracy.
- `freeze_backbone`, `replace_classifier`, and `unfreeze_all`: explicit
  transfer-learning helpers.

## Optional real CIFAR-10 run

Install a compatible `torchvision` separately, then opt in explicitly:

```python
from torch.utils.data import DataLoader
from cifar_resnet import CIFARTrainTransform, cifar10_datasets, resnet18
from cifar_resnet.train import fit

train, test = cifar10_datasets(
    root="data/cifar10", train_transform=CIFARTrainTransform(), download=True
)
model = resnet18()
history = fit(model, DataLoader(train, batch_size=128, shuffle=True), epochs=90)
```

The repository tests do not execute this path and do not require a network.
For a real experiment, add normalization, a validation split, a scheduler,
checkpointing, and a fixed run manifest before interpreting the score.

## What to inspect

1. Follow one `BasicBlock` forward pass and write down the two possible skip
   paths. The projection branch is required when stride or channel count
   changes.
2. Compare a tiny `CIFARCNN` and `resnet18` at the same parameter budget. Test
   shape first, then inspect gradients after one loss backward pass.
3. Seed two `CIFARTrainTransform` calls with separate generators and verify
   identical crops. Data-loader worker scheduling should not silently become a
   source of irreproducibility.
4. Freeze a ResNet backbone, replace `fc`, and check that only the new head is
   trainable before a transfer-learning run.

## Honest boundaries

The fake fixture checks tensor plumbing, skip connections, gradients, and an
offline training loop. It is not evidence of CIFAR-10 accuracy. A CIFAR-10
ResNet-18 from scratch reaching roughly **90% locally on Apple MPS is
realistic** with a carefully tuned run; **93%+ generally needs cloud GPU time
or MLX/longer experiments**. Exact results depend on seed, augmentation,
normalization, scheduler, and hardware. No benchmark claim should be made from
the synthetic tests.
