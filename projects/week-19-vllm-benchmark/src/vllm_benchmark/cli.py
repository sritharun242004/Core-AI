"""python -m vllm_benchmark demo | benchmark --engine hf|vllm --model-path ..."""

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from .benchmark import (
    HFAdapter,
    LocalModelConfig,
    VLLMAdapter,
    environment_metadata,
    run_benchmark,
)
from .cache import PagedKVCache
from .metrics import RequestTrace, aggregate_metrics
from .quantization import dequantize, quantize
from .speculation import acceptance_probability, residual_distribution


class DeviceProperties(Protocol):
    total_memory: int


class CudaRuntime(Protocol):
    def get_device_name(self, index: int) -> str: ...

    def get_device_properties(self, index: int) -> DeviceProperties: ...


class TorchVersion(Protocol):
    cuda: str | None


class TorchRuntime(Protocol):
    cuda: CudaRuntime
    version: TorchVersion


def offline_demo() -> dict[str, object]:
    weights = np.array([[-2.0, 0.3, 0.0, 0.8, 7.0]], dtype=np.float32)
    quantized = quantize(weights, bits=4, group_size=2)
    cache = PagedKVCache(3, 4)
    cache.allocate("a", 5)
    cache.allocate("b", 4)
    waste = cache.internal_waste_tokens
    cache.release("b")
    cache.check_invariants()
    traces = [RequestTrace(0, (0.2, 0.3, 0.4), 0.5), RequestTrace(0.1, (0.6,), 0.8)]
    target, draft = [0.6, 0.3, 0.1], [0.2, 0.3, 0.5]
    return {
        "evidence": "simulation_not_hardware_measurement",
        "quantization": {
            "bits": 4,
            "codes": quantized.codes.tolist(),
            "scales": quantized.scales.tolist(),
            "max_error": float(np.max(np.abs(dequantize(quantized) - weights))),
            "estimated_packed_bytes": quantized.estimated_packed_bytes,
            "storage_note": "INT4 codes are stored unpacked in int8 arrays",
        },
        "paging": {"tail_waste_before_free": waste, "free_pages_after_release": cache.free_pages},
        "speculation": {
            "accept": acceptance_probability(target, draft).tolist(),
            "residual": residual_distribution(target, draft).tolist(),
        },
        "metrics": aggregate_metrics(traces, window_start_s=0, window_end_s=1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="offline arithmetic; NOT a hardware benchmark")
    demo.add_argument("--output", type=Path)
    actual = sub.add_parser("benchmark", help="measure one explicit local model; no downloads")
    actual.add_argument("--engine", choices=("hf", "vllm"), required=True)
    actual.add_argument("--model-path", type=Path, required=True)
    actual.add_argument(
        "--revision-label", required=True, help="checkpoint SHA or immutable manifest ID"
    )
    actual.add_argument(
        "--prompts-file", type=Path, required=True, help="local JSON array of strings"
    )
    actual.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    actual.add_argument("--max-new-tokens", type=int, default=32)
    actual.add_argument("--max-model-len", type=int, default=1024)
    actual.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    actual.add_argument(
        "--prefix-cache", action="store_true", help="vLLM only; warm cache after warmup"
    )
    actual.add_argument("--repeats", type=int, default=3)
    actual.add_argument("--warmup", type=int, default=1)
    actual.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            report = offline_demo()
        else:
            config = LocalModelConfig(
                model_path=args.model_path,
                revision_label=args.revision_label,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                prefix_cache=args.prefix_cache,
            )
            prompts_value = json.loads(args.prompts_file.read_text())
            if not isinstance(prompts_value, list):
                raise ValueError("prompts file must contain a nonempty JSON array of strings")
            prompt_values = cast(list[object], prompts_value)
            if not prompt_values or any(
                not isinstance(p, str) or not p.strip() for p in prompt_values
            ):
                raise ValueError("prompts file must contain a nonempty JSON array of strings")
            prompts = cast(list[str], prompt_values)
            # Validate workload options before loading expensive weights.
            from .quantization import positive_int

            positive_int(args.repeats, "repeats")
            positive_int(args.warmup, "warmup", allow_zero=True)
            adapter = HFAdapter(config) if args.engine == "hf" else VLLMAdapter(config)
            report = run_benchmark(
                adapter,
                prompts,
                repeats=args.repeats,
                warmup=args.warmup,
                max_total_tokens=config.max_model_len,
                max_new_tokens=config.max_new_tokens,
            )
            report["engine"] = args.engine
            report["config"] = {**asdict(config), "model_path": str(config.model_path)}
            report["environment"] = environment_metadata()
            report["instrumentation"] = adapter.metadata
            if config.device == "cuda":
                torch = cast(TorchRuntime, importlib.import_module("torch"))
                environment = report["environment"]
                environment["gpu"] = torch.cuda.get_device_name(0)
                environment["cuda"] = torch.version.cuda
                environment["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Benchmark aborted (no successful report): {exc}", file=sys.stderr)
        return 1
    return 0
