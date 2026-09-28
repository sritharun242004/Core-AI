# Challenge — transfer and ablation (3+ hrs)

1. Load a compatible pretrained image backbone (outside the reference test
   dependency set), replace its classifier for ten classes, and freeze the
   backbone. Verify optimizer parameter groups contain only the new head.
2. Unfreeze the final residual stage with a lower learning rate and compare
   against head-only training.
3. Ablate one choice: remove augmentation, replace zero-initialized residual
   BatchNorm, change the stem to an ImageNet-style stem, or swap SGD for AdamW.
4. Add a regression test for the behavior you want to preserve. Include one
   negative test for a bad input shape or invalid hyperparameter.

**Deliverable:** an ablation report with curves, parameter counts, and a short
failure analysis. Explain whether the change affected optimization, capacity,
or the data prior; do not infer causality from one seed.
