"""Optional real measurements. Imports never load torch, Transformers, or vLLM.

Adapters use only explicit local directories; the baseline tests do not execute a
model. These are instrumented concurrency=1 measurements, not serving load tests.
"""

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .metrics import RequestTrace, aggregate_metrics
from .quantization import positive_int


class TorchTensor(Protocol):
    def numel(self) -> int: ...


class TorchCuda(Protocol):
    def synchronize(self) -> None: ...


class TorchAPI(Protocol):
    float32: object
    float16: object
    cuda: TorchCuda

    def tensor(self, data: list[list[int]], *, device: str) -> TorchTensor: ...

    def ones_like(self, tensor: TorchTensor) -> TorchTensor: ...

    def inference_mode(self) -> AbstractContextManager[object]: ...


class TokenStreamer(Protocol):
    def put(self, values: TorchTensor) -> None: ...

    def end(self) -> None: ...


class Tokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]: ...


class CausalModel(Protocol):
    def to(self, device: str) -> "CausalModel": ...

    def eval(self) -> "CausalModel": ...

    def generate(
        self,
        *,
        input_ids: TorchTensor,
        attention_mask: TorchTensor,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        use_cache: bool,
        streamer: TokenStreamer,
        pad_token_id: int | None,
    ) -> object: ...


class HFTokenizerFactory(Protocol):
    def from_pretrained(
        self, model_path: str, *, local_files_only: bool, trust_remote_code: bool
    ) -> Tokenizer: ...


class HFModelFactory(Protocol):
    def from_pretrained(
        self,
        model_path: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: object,
    ) -> CausalModel: ...


class EngineResultOutput(Protocol):
    token_ids: Sequence[int]


class EngineResult(Protocol):
    request_id: str
    outputs: Sequence[EngineResultOutput]
    finished: bool


class Engine(Protocol):
    def add_request(self, request_id: str, prompt: object, params: object) -> None: ...

    def has_unfinished_requests(self) -> bool: ...

    def step(self) -> Sequence[EngineResult]: ...


class EngineClass(Protocol):
    @classmethod
    def from_engine_args(cls, args: object) -> Engine: ...


@dataclass(frozen=True)
class LocalModelConfig:
    model_path: Path
    revision_label: str
    max_new_tokens: int = 32
    max_model_len: int = 1024
    device: str = "cpu"
    gpu_memory_utilization: float = 0.8
    prefix_cache: bool = False

    def __post_init__(self) -> None:
        path = Path(self.model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                "an existing local model directory is required; no model IDs/downloads"
            )
        object.__setattr__(self, "model_path", path)
        if not self.revision_label.strip():
            raise ValueError("revision_label must identify the local checkpoint")
        positive_int(self.max_new_tokens, "max_new_tokens")
        positive_int(self.max_model_len, "max_model_len")
        if self.max_new_tokens >= self.max_model_len:
            raise ValueError("max_new_tokens must leave room for a prompt")
        if (
            not math.isfinite(self.gpu_memory_utilization)
            or not 0 < self.gpu_memory_utilization < 1
        ):
            raise ValueError("gpu_memory_utilization must lie strictly between 0 and 1")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")


def _offline_environment() -> None:
    # Set before lazy imports, even if the caller inherited an online setting.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"


class Adapter(Protocol):
    def encode(self, prompt: str) -> list[int]: ...
    def generate(self, token_ids: list[int]) -> RequestTrace: ...


class HFAdapter:
    """Greedy .generate with a token-ID streamer, synchronized before each timestamp."""

    def __init__(self, config: LocalModelConfig):
        _offline_environment()
        torch = cast(TorchAPI, importlib.import_module("torch"))
        transformers = importlib.import_module("transformers")
        model_factory = cast(HFModelFactory, transformers.AutoModelForCausalLM)
        tokenizer_factory = cast(HFTokenizerFactory, transformers.AutoTokenizer)

        if config.prefix_cache:
            raise ValueError("this HF adapter has no cross-request prefix cache")
        self.config = config
        self.torch: TorchAPI = torch
        self.tokenizer: Tokenizer = tokenizer_factory.from_pretrained(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        dtype = torch.float32 if config.device == "cpu" else torch.float16
        self.model: CausalModel = (
            model_factory.from_pretrained(
                str(config.model_path),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
            )
            .to(config.device)
            .eval()
        )
        self.metadata: dict[str, object] = {
            "dtype": str(dtype),
            "token_observation": "synchronized token-ID streamer",
        }

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def _sync(self) -> None:
        if self.config.device == "cuda":
            self.torch.cuda.synchronize()

    def generate(self, token_ids: list[int]) -> RequestTrace:
        token_times: list[float] = []
        sync = self._sync

        class TokenClock:
            prompt_pending = True

            def put(self, values: TorchTensor) -> None:
                if self.prompt_pending:
                    self.prompt_pending = False
                    return  # generate first passes input IDs, which are NOT output tokens.
                sync()
                token_times.extend([time.perf_counter()] * values.numel())

            def end(self) -> None:
                pass

        self._sync()
        submitted = time.perf_counter()
        inputs = self.torch.tensor([token_ids], device=self.config.device)
        with self.torch.inference_mode():
            self.model.generate(
                input_ids=inputs,
                attention_mask=self.torch.ones_like(inputs),
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                streamer=TokenClock(),
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        self._sync()
        return RequestTrace(submitted, tuple(token_times), time.perf_counter())


class VLLMAdapter:
    """LLMEngine step loop; cumulative output deltas get engine-observation timestamps.

    If an engine step returns multiple tokens, their timestamps coincide. This is
    intentionally reported as coalesced observation time, not per-token kernel time.
    """

    def __init__(self, config: LocalModelConfig):
        if config.device != "cuda":
            raise ValueError("this optional vLLM adapter targets a CUDA host; select --device cuda")
        _offline_environment()
        transformers = importlib.import_module("transformers")
        vllm = importlib.import_module("vllm")
        tokenizer_factory = cast(HFTokenizerFactory, transformers.AutoTokenizer)
        engine_args_factory = cast(Callable[..., object], vllm.EngineArgs)
        engine_class = cast(EngineClass, vllm.LLMEngine)
        sampling_params_factory = cast(Callable[..., object], vllm.SamplingParams)

        self.config = config
        self.sampling_params = sampling_params_factory(
            temperature=0, max_tokens=config.max_new_tokens, seed=19
        )
        self.tokenizer: Tokenizer = tokenizer_factory.from_pretrained(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        args = engine_args_factory(
            model=str(config.model_path),
            tokenizer=str(config.model_path),
            trust_remote_code=False,
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.prefix_cache,
            disable_log_stats=True,
            enforce_eager=True,
            seed=19,
        )
        self.engine: Engine = engine_class.from_engine_args(args)
        self._next_id = 0
        self.metadata: dict[str, object] = {
            "dtype": "float16",
            "token_observation": "coalesced engine-step deltas",
            "enforce_eager": True,
            "tensor_parallel_size": 1,
        }

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def generate(self, token_ids: list[int]) -> RequestTrace:
        self._next_id += 1
        request_id = str(self._next_id)
        token_times: list[float] = []
        submitted = time.perf_counter()
        self.engine.add_request(request_id, {"prompt_token_ids": token_ids}, self.sampling_params)
        finished = False
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            observed = time.perf_counter()
            for result in outputs:
                if result.request_id != request_id:
                    continue
                if result.outputs:
                    count = len(result.outputs[0].token_ids)
                    if count < len(token_times):
                        raise RuntimeError("vLLM outputs are not cumulative; incompatible API")
                    token_times.extend([observed] * (count - len(token_times)))
                finished = bool(result.finished)
        if not finished:
            raise RuntimeError("vLLM request did not finish successfully")
        return RequestTrace(submitted, tuple(token_times), time.perf_counter())


def run_benchmark(
    adapter: Adapter | None,
    prompts: Sequence[object],
    *,
    repeats: int = 3,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    max_total_tokens: int | None = None,
    max_new_tokens: int = 32,
) -> dict[str, object]:
    """Pretokenized serial workload; excludes load/tokenization/warmup from wall window.

    All errors propagate: never produce a deceptively successful partial summary.
    Clock and adapter injection make instrumentation testable without engine packages.
    """
    positive_int(repeats, "repeats")
    positive_int(warmup, "warmup", allow_zero=True)
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("prompts must be a nonempty list of nonempty strings")
    if adapter is None:
        raise ValueError("adapter is required for a nonempty workload")
    valid_prompts = cast(list[str], list(prompts))
    encoded = [adapter.encode(prompt) for prompt in valid_prompts]
    if any(not ids for ids in encoded):
        raise ValueError("a prompt tokenized to zero tokens")
    if max_total_tokens is not None and any(
        len(ids) + max_new_tokens > max_total_tokens for ids in encoded
    ):
        raise ValueError("prompt + requested output exceeds max_model_len; no silent truncation")
    for index in range(warmup):
        adapter.generate(encoded[index % len(encoded)])
    start = clock()
    traces = [adapter.generate(ids) for _ in range(repeats) for ids in encoded]
    end = clock()
    return {
        "evidence": "instrumented_local_model_measurement",
        "workload": {
            "concurrency": 1,
            "repeats": repeats,
            "warmup_requests": warmup,
            "input_tokens": [len(ids) for ids in encoded],
            "prompt_token_sha256": hashlib.sha256(json.dumps(encoded).encode()).hexdigest(),
            "scope": "pretokenized serial requests; excludes load/tokenization/warmup",
        },
        "metrics": aggregate_metrics(traces, window_start_s=start, window_end_s=end),
        "requests": [
            {
                "output_tokens": len(t.token_times_s),
                "ttft_s": t.ttft_s,
                "tpot_s": t.tpot_s,
                "latency_s": t.latency_s,
                "token_offsets_s": [v - t.submitted_s for v in t.token_times_s],
            }
            for t in traces
        ],
    }


def environment_metadata() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
