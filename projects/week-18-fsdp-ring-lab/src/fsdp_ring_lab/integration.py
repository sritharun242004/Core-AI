"""Optional real CUDA torchrun smoke test. Never used by the offline notebook.

FSDP1 FULL_SHARD is used explicitly, not the different FSDP2 fully_shard API.
DeepSpeed is a lazy, optional import installed only in a compatible GPU image.
Both backends compare tiny FP32 Adam updates with a full-batch oracle. This is
correctness smoke testing, not evidence of memory savings or training speed.
"""

import argparse
import copy
import json
import os
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from ._torch import backward, manual_seed


def check_launch(
    environment: Mapping[str, str], *, cuda_available: bool, nccl_available: bool, device_count: int
) -> tuple[int, int, int]:
    """Pure preflight: refuse CPU/MPS and malformed/non-torchrun launches."""
    if not cuda_available or not nccl_available:
        raise RuntimeError(
            "Real integration requires CUDA GPUs and a PyTorch NCCL build; "
            "use the CPU notebook instead"
        )
    try:
        rank, local_rank, world = (
            int(environment[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "Launch with torchrun; RANK, LOCAL_RANK, WORLD_SIZE must be integers"
        ) from error
    if world < 2 or not 0 <= rank < world or not 0 <= local_rank < device_count:
        raise RuntimeError("Need at least two ranks and one visible CUDA device per local rank")
    return rank, local_rank, world


def load_zero_config(path: Path, *, batch_size: int) -> dict:
    config = json.loads(path.read_text())
    if (
        config.get("train_micro_batch_size_per_gpu") != batch_size
        or config.get("gradient_accumulation_steps") != 1
        or config.get("zero_optimization", {}).get("stage") not in (2, 3)
        or config.get("fp16", {}).get("enabled", False)
        or config.get("bf16", {}).get("enabled", False)
        or "optimizer" in config
        or "train_batch_size" in config
    ):
        raise ValueError(
            "Smoke config needs matching microbatch, accumulation=1, stage 2/3, FP32, "
            "and no optimizer/global-batch override"
        )
    return config


def synthetic_batch(
    rank: int, step: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1800 + step * 10000 + rank)
    x = torch.randn(batch_size, 8, generator=generator).to(device)
    target = torch.randn(batch_size, 2, generator=generator).to(device)
    return x, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("fsdp", "deepspeed"), default="fsdp")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--zero-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "zero-stage3.json",
    )
    # Some DeepSpeed launchers pass this flag; torchrun also sets LOCAL_RANK.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    rank, local_rank, world = check_launch(
        os.environ,
        cuda_available=torch.cuda.is_available(),
        nccl_available=dist.is_nccl_available(),
        device_count=torch.cuda.device_count(),
    )
    config = (
        load_zero_config(args.zero_config, batch_size=args.batch_size)
        if args.engine == "deepspeed"
        else None
    )
    if args.engine == "deepspeed":
        try:
            import deepspeed
        except ImportError as error:
            raise RuntimeError(
                "DeepSpeed is optional; install it in a compatible CUDA image, "
                "not the offline workspace"
            ) from error
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    from torch.distributed.fsdp import ShardingStrategy

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    manual_seed(18)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        base = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 2)).to(device)
        reference = copy.deepcopy(base)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
        if args.engine == "fsdp":
            model = FSDP(
                base,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=device,
                use_orig_params=True,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            model, optimizer, _, _ = deepspeed.initialize(
                model=base,
                optimizer=torch.optim.Adam(base.parameters(), lr=1e-3),
                config=config,
                dist_init_required=False,
            )
        for step in range(args.steps):
            batches = [synthetic_batch(r, step, args.batch_size, device) for r in range(world)]
            x_all = torch.cat([batch[0] for batch in batches])
            y_all = torch.cat([batch[1] for batch in batches])
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = (reference(x_all) - y_all).square().mean()
            backward(reference_loss)
            reference_optimizer.step()
            x, target = batches[rank]
            if args.engine == "fsdp":
                optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            if args.engine == "fsdp":
                backward(loss)
                optimizer.step()
                context = FSDP.summon_full_params(model, writeback=False)
            else:
                model.backward(loss)
                model.step()
                context = deepspeed.zero.GatheredParameters(
                    list(base.parameters()), modifier_rank=None
                )
            averaged_loss = loss.detach().clone()
            dist.all_reduce(averaged_loss)
            averaged_loss /= world
            torch.testing.assert_close(averaged_loss, reference_loss.detach(), atol=1e-5, rtol=1e-4)
            with context:
                for actual, expected in zip(base.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "engine": args.engine,
                            "step": step,
                            "world_size": world,
                            "global_batch": world * args.batch_size,
                            "mean_loss": averaged_loss.item(),
                            "reference_weights_match": True,
                            "torch": torch.__version__,
                            "device": torch.cuda.get_device_name(local_rank),
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
