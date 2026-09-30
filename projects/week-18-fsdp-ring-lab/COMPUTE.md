# Compute plan — 🔴 red distributed extension

## Zero-cost baseline

The notebook and all pytest tests run on CPU (including Apple Silicon), use tiny
synthetic tensors, and require no API key, model, dataset, or cloud service. Expect
seconds to minutes depending on hardware and startup; this is an estimate, not a
benchmark. MPS is not a substitute for the CUDA/NCCL torchrun integration.

## Provider, hours, and stop-loss

- **Suggested provider:** RunPod, on-demand pod, one node with **2 × A100 40 GB** and
  an explicitly checked inter-GPU topology. A two-GPU alternative is acceptable only
  after checking memory, NCCL support, permissions, and connectivity. Two isolated
  one-GPU pods are not the same experiment.
- **Session estimate:** 1–2 hours including environment checks and three smoke runs.
  The tiny updates themselves should be far shorter; setup and debugging dominate.
- **Planning rate only:** assume **$2–$5 per GPU-hour**, not a live quote. Two GPUs for
  two hours imply **$8–$20 compute**, plus storage, tax, and any egress.
- **Hard learner budget:** approve no more than **$25 all-in** for the first session.
  Check the provider's current **whole-pod** quote before booking; do not multiply a
  whole-node price by the GPU count again. Set a two-hour stop alarm and a provider
  spend alert. If the quote exceeds the budget, stop and use the CPU path.
- No credit card, provisioning, cloud launch, or paid execution is performed by this
  repository. Prices and availability are estimates, never guarantees.

## Before starting the billing clock

1. Pass CPU tests and notebook locally. Save the commit/content version and expected
   numerical contracts. Choose FSDP first; DeepSpeed is an additional compatibility task.
2. Select a trusted CUDA/PyTorch image with Python 3.13 support. Confirm its installed
   PyTorch/CUDA/driver versions and NCCL backend, disk costs, region, and GPU count.
   DeepSpeed's optional compiled extensions must support that image; do not change the
   shared monorepo lockfile to experiment. Keep any installation inside the rented image.
3. Prepare this small source project only. No checkpoint or training data is needed.
   Do not paste credentials into notebooks, git files, logs, or shell history.
4. Configure billing alerts and write down the actual pod id and planned termination
   time. A notebook timeout or killing Python **does not** stop infrastructure charges.

## On the approved pod

```bash
nvidia-smi
nvidia-smi topo -m
python -c 'import torch; import torch.distributed as d; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), d.is_nccl_available())'
# Expect at least two visible CUDA GPUs and NCCL support, then from the project:
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m fsdp_ring_lab.integration --engine fsdp --steps 3
```

Only after that succeeds, follow the two DeepSpeed commands in README with stage-2 and
stage-3 JSON. Do not launch all experiments concurrently. Confirm every step emits a
rank-0 JSON record with `reference_weights_match: true`. A two-rank launch checks real
synchronization; a single-device run cannot establish distributed correctness. If NCCL
fails, retain diagnostic logs, stop the job, and inspect topology/version compatibility;
do not disable checks or leave a paid pod idle while investigating locally.

## What to record, and what not to claim

Record provider/region, GPU model and count, NVLink/PCIe topology, drivers, Python,
PyTorch/CUDA/NCCL/DeepSpeed versions, backend, config, seed, dtype, global/microbatch,
steps, elapsed session time, loss/weight comparisons, quoted and actual invoice cost.
`integration.py` intentionally has a full reference model on each rank and is not a
memory benchmark. It provides no throughput or speedup number. Analytic byte counts
omit temporary buffers; the CPU ring simulator has no real communication overlap.

For a later **separate** benchmark, remove the duplicated oracle after correctness is
established, warm up, synchronize CUDA before/after timing, report repeated timings and
maximum **across ranks**, sample `max_memory_allocated` and reserved memory, fix model,
dtype, effective batch and token count, and disclose checkpoint/prefetch policies. GPU
utilization alone does not measure useful throughput. Do not extrapolate a tiny MLP or
CPU block loop to a frontier-model cluster.

## Mandatory teardown — even after failure

1. Stop torchrun and confirm no worker remains (`nvidia-smi` and process list).
2. Copy only small logs/config/version records you need; do not create persistent
   volumes to save a three-step synthetic model.
3. **Terminate/delete the RunPod pod** in the provider console. Merely stopping a
   container, disconnecting SSH, closing Jupyter, or exiting Python is insufficient.
4. Delete attached paid volumes/snapshots and unused network volumes after explicitly
   confirming they contain no wanted work. Persistent storage can bill after compute stops.
5. Verify no active GPU pod and no unintended paid volume remain, check the billing
   page for the final charge, and save the receipt in your experiment notes.
6. Revoke temporary credentials and remove SSH keys from the rented environment.

For Modal or vast.ai instead, first translate the same checklist to their exact
resource lifecycle and current billing terms; this document does not guarantee a
provider-independent stop command. When in doubt, use the console and verify billing.
