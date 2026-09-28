# Compute note

**Tier:** 🟡 local MPS for a serious baseline · **tests:** 🟢 CPU/offline ·
**budget for tests:** $0

The project tests run on CPU, use only deterministic in-memory fake data, and
never download CIFAR-10. A tiny ResNet smoke test should finish in seconds to
a few minutes depending on the machine. PyTorch is the only required runtime
dependency; `torchvision` is optional and is imported only for an explicitly
requested real-data run.

## Realistic expectation

A tuned ResNet-18 from scratch can reach **roughly 90% on CIFAR-10 locally on
Apple MPS** in about an hour on a recent Apple Silicon laptop. **93%+ generally
needs cloud GPU time or MLX/overnight experiments**; it is not a result promised
by the fake-data tests. Factors include seed, normalization, augmentation,
learning-rate schedule, batch size, PyTorch version, and thermal throttling.
Record all of them with the result.

## Suggested local run

1. Install a PyTorch build with MPS support and a compatible torchvision only if
   using the real loader.
2. Download once with `cifar10_datasets(..., download=True)` to
   `data/cifar10/`; do not put downloaded data in git.
3. Use `device=choose_device()` (or `device="mps"`), batch size 128 if memory
   permits, normalization from the training split, random crop/flip, SGD with
   momentum, and a cosine or step schedule.
4. Save checkpoints and a JSON manifest containing seed, device, epochs,
   transforms, parameter count, and train/validation/test metrics.

## Cloud/MLX upgrade path

For 93%+ experiments, use an explicitly budgeted cloud GPU (for example an
A10G or A100 through RunPod, Modal, or a comparable provider), or port the
experiment to MLX on Apple Silicon. Set a spend cap, use a resumable
checkpoint, and stop the instance after the run. This educational repository
does not automate cloud provisioning and does not include credentials.
