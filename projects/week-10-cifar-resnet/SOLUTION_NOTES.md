# Solution notes — CNNs and residual networks

## 1. Derive the residual block before coding it

A plain layer learns a transformation `F(x)`. A residual block instead returns

`y = ReLU(F(x) + S(x))`,

where `S(x) = x` when channels and spatial resolution already match. If a
stage doubles channels or uses stride 2, `S` is a learned 1×1 convolution plus
batch normalization. Addition is only valid after both paths have exactly the
same shape; this is why `BasicBlock.downsample` is either `None` or an explicit
projection.

The practical benefit is an identity route for signal and gradients. It does
not make a network automatically good: normalization, initialization,
augmentation, schedule, and data quality still matter. This implementation
zero-initializes the second batch-normalization scale in each block, so a new
block begins close to an identity mapping (after the final ReLU).

## 2. Why CIFAR uses a different stem

ImageNet ResNet-18 commonly starts with a 7×7 stride-2 convolution and max
pool. On a 32×32 CIFAR image that discards too much spatial information. The
project starts with a 3×3 stride-1 convolution and no max pool, then downsamples
at later residual stages. Global average pooling removes a fragile flatten-size
calculation before the classifier.

## 3. Augmentation is a training-only prior

Random crop with padding and horizontal flip create plausible variants without
changing the test distribution. The tensor-only transform accepts a
`torch.Generator`; tests can therefore prove deterministic behavior without
relying on global RNG state. Do not flip classes for which left/right carries
meaning. Normalize using training-set statistics and apply the same fixed
normalization to validation and test images.

## 4. What the tests establish

- model and block outputs retain the batch and class dimensions;
- identity and projection skips take the correct shape path;
- a scalar loss reaches convolution, batch-normalization, and classifier
  parameters with finite gradients;
- fake samples, labels, and seeded transforms are deterministic;
- a tiny CPU training smoke test returns finite metrics.

These are implementation contracts, not a claim that the fake data is CIFAR or
that two epochs are a useful benchmark.

## 5. Transfer learning boundary

For a pretrained backbone, `freeze_backbone` marks every parameter except a
model's `fc` or `classifier` head as frozen. `replace_classifier` swaps that
head with a fresh `Linear` layer. In a real run, train the head first, then
unfreeze selected stages with a lower learning rate. BatchNorm needs an
explicit policy: freezing weights alone does not stop running statistics from
changing while the model remains in training mode.

## 6. Accuracy and compute claims

A careful local MPS ResNet-18 experiment can reach roughly 90% CIFAR-10, but it
needs a real data pipeline, normalization, augmentation, a learning-rate
schedule, and enough epochs. 93%+ is not a promise of this reference project;
it usually needs cloud GPU time or MLX and longer tuning. The offline suite
intentionally avoids all downloads and uses CPU-sized models so tests stay
fast, deterministic, and reviewable.

## Debugging checklist

- A shape error at `out + identity` means the projection condition omitted a
  stride or channel mismatch.
- A score stuck near random chance may be label/normalization/device trouble,
  not a reason to add depth blindly.
- If train loss is reported incorrectly for a final short batch, aggregate
  `loss * batch_size`, then divide by total examples (as `train_epoch` does).
- If augmentation tests flicker, pass a local generator and make worker seeds
  explicit; do not patch tests with a global random seed.
- If transfer learning changes the frozen backbone, inspect
  `requires_grad`, optimizer parameter groups, and BatchNorm train/eval mode.
