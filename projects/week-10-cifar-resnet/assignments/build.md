# Build — an honest CIFAR experiment (2–3 hrs)

1. Compare `CIFARCNN` and `resnet18` at a deliberately small width. Record
   parameter counts, tensor shapes, and one-epoch CPU metrics on
   `make_fake_cifar`.
2. Add a real CIFAR-10 data path using optional torchvision. Keep download and
   tests separate: no test may contact the network.
3. Add train-only random crop/flip and train-set normalization. Keep
   validation/test transforms deterministic.
4. Train a real CIFAR baseline on MPS if available. Log seed, device, batch
   size, schedule, augmentation, and wall-clock time.

**Deliverable:** a one-page experiment table with at least two runs and a
paragraph that distinguishes synthetic smoke-test results from real CIFAR
results. Treat roughly 90% local MPS as a realistic target; do not present
93%+ as a laptop guarantee.
