# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnnecessaryCast=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportOptionalOperand=false
"""CLI integration tests — construct fake vllm/HF adapter instances via object.__new__
and attribute injection; typed assignment is intentionally bypassed in test fixtures."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest
from vllm_benchmark.benchmark import (
    Adapter,
    Engine,
    EngineResult,
    HFAdapter,
    LocalModelConfig,
    VLLMAdapter,
    run_benchmark,
)
from vllm_benchmark.metrics import RequestTrace


class ConfigOverrides(TypedDict, total=False):
    max_new_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    revision_label: str


class ConfigValues(TypedDict):
    model_path: Path
    revision_label: str
    max_new_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    prefix_cache: bool


def test_import_does_not_load_optional_engines() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_benchmark.benchmark; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
            "assert 'vllm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_demo_cli_is_offline_and_labels_simulation(tmp_path: Path) -> None:
    output = tmp_path / "demo.json"
    result = subprocess.run(
        [sys.executable, "-m", "vllm_benchmark", "demo", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["evidence"] == "simulation_not_hardware_measurement"
    assert report["metrics"]["output_tokens_per_s"] == 4


def test_nonexistent_local_model_rejected_before_optional_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local model"):
        LocalModelConfig(model_path=tmp_path / "missing", revision_label="sha")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_model_len": -1},
        {"gpu_memory_utilization": 1.1},
        {"revision_label": ""},
    ],
)
def test_invalid_local_model_config(tmp_path: Path, kwargs: ConfigOverrides) -> None:
    values: ConfigValues = {
        "model_path": tmp_path,
        "revision_label": "sha",
        "max_new_tokens": 32,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.8,
        "prefix_cache": False,
    }
    values = cast(ConfigValues, {**values, **kwargs})
    with pytest.raises(ValueError):
        LocalModelConfig(**values)


def test_real_cli_requires_explicit_model_and_prompts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vllm_benchmark", "benchmark", "--engine", "hf"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--model-path" in result.stderr


def test_harness_excludes_warmup_reports_serial_scope_and_metadata(tmp_path: Path) -> None:
    class FakeAdapter:
        calls = 0

        def generate(self, token_ids: list[int]) -> RequestTrace:
            self.calls += 1
            return RequestTrace(10 + self.calls, (10.2 + self.calls,), 10.5 + self.calls)

        def encode(self, prompt: str) -> list[int]:
            return [1, 2]

    adapter = FakeAdapter()
    ticks = iter([12.0, 14.0])  # Warmup call=1; measured calls=2,3 in [12,14].
    report = run_benchmark(
        adapter, ["private prompt"], repeats=2, warmup=1, clock=lambda: next(ticks)
    )
    assert adapter.calls == 3
    metrics = cast(dict[str, object], report["metrics"])
    workload = cast(dict[str, object], report["workload"])
    assert metrics["output_tokens_per_s"] == 1.0
    assert metrics["completed_requests"] == 2
    assert workload["concurrency"] == 1
    assert workload["warmup_requests"] == 1
    assert "private prompt" not in json.dumps(report)


def test_empty_prompt_and_invalid_repeats_rejected() -> None:
    with pytest.raises(ValueError):
        run_benchmark(None, [], repeats=1, warmup=0)
    with pytest.raises(ValueError):
        run_benchmark(None, ["x"], repeats=0, warmup=0)


def test_token_budget_rejected_before_generation() -> None:
    def encode(_: str) -> list[int]:
        return [1, 2, 3]

    adapter = cast(Adapter, SimpleNamespace(encode=encode))
    with pytest.raises(ValueError, match="max_model_len"):
        run_benchmark(adapter, ["x"], max_total_tokens=4, max_new_tokens=2)


def test_vllm_observes_cumulative_deltas_not_recounted_tokens() -> None:
    class FakeEngine:
        steps = 0

        def add_request(self, request_id: str, prompt: object, params: object) -> None:
            assert prompt == {"prompt_token_ids": [7, 8]}

        def has_unfinished_requests(self) -> bool:
            return self.steps < 2

        def step(self) -> list[EngineResult]:
            self.steps += 1
            ids = [3] if self.steps == 1 else [3, 4, 5]
            return cast(
                list[EngineResult],
                [
                    SimpleNamespace(
                        request_id="1",
                        finished=self.steps == 2,
                        outputs=[SimpleNamespace(token_ids=ids)],
                    )
                ],
            )

    adapter = cast(VLLMAdapter, object.__new__(VLLMAdapter))
    adapter.engine = cast(Engine, FakeEngine())
    adapter._next_id = 0
    adapter.sampling_params = object()
    trace = adapter.generate([7, 8])
    assert len(trace.token_times_s) == 3
    assert trace.token_times_s[1] == trace.token_times_s[2]  # Coalesced engine observation.
    assert trace.ttft_s is not None and trace.ttft_s >= 0


def test_hf_streamer_skips_prompt_and_counts_token_ids(tmp_path: Path) -> None:
    from contextlib import nullcontext

    def generate(**kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["pad_token_id"] == 0  # Zero is a valid token ID, not a missing value.
        stream = kwargs["streamer"]
        stream.put(SimpleNamespace(numel=lambda: 6))  # Prompt; must NOT be counted.
        stream.put(SimpleNamespace(numel=lambda: 1))
        stream.put(SimpleNamespace(numel=lambda: 1))
        stream.end()

    adapter = object.__new__(HFAdapter)
    adapter.config = LocalModelConfig(tmp_path, "sha")
    adapter.torch = SimpleNamespace(
        tensor=lambda *a, **kw: object(), ones_like=lambda _: object(), inference_mode=nullcontext
    )
    adapter.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    adapter.model = SimpleNamespace(generate=generate)
    trace = adapter.generate([1, 2, 3, 4, 5, 6])
    assert len(trace.token_times_s) == 2
    assert trace.tpot_s >= 0
