# Compute note

**Tier:** 🟡 local stretch · **Toy path:** $0, CPU/MPS, 1–5 minutes · **Optional scale path:** approximately $5–15 for a small single-GPU run.

All automated tests and the notebook use six short in-memory stories and a model under a few million parameters. No network, account, cloud GPU, or pretrained weights are required. For a real TinyStories pretraining experiment, use a provider such as RunPod or Modal, pin the dataset/checkpoint versions, record hours and token count, and stop the instance after the run. Do not mistake the toy loss curve for a scaling benchmark.
