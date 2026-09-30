import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm_benchmark.benchmark import HFAdapter, LocalModelConfig, VLLMAdapter, run_benchmark
from vllm_benchmark.metrics import RequestTrace


def test_import_does_not_load_optional_engines():
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


def test_demo_cli_is_offline_and_labels_simulation(tmp_path):
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


def test_nonexistent_local_model_rejected_before_optional_import(tmp_path):
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
def test_invalid_local_model_config(tmp_path, kwargs):
    values = dict(model_path=tmp_path, revision_label="sha")
    values.update(kwargs)
    with pytest.raises(ValueError):
        LocalModelConfig(**values)


def test_real_cli_requires_explicit_model_and_prompts():
    result = subprocess.run(
        [sys.executable, "-m", "vllm_benchmark", "benchmark", "--engine", "hf"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--model-path" in result.stderr


def test_harness_excludes_warmup_reports_serial_scope_and_metadata(tmp_path):
    class FakeAdapter:
        calls = 0

        def generate(self, token_ids):
            self.calls += 1
            return RequestTrace(10 + self.calls, (10.2 + self.calls,), 10.5 + self.calls)

        def encode(self, prompt):
            return [1, 2]

    adapter = FakeAdapter()
    ticks = iter([12.0, 14.0])  # Warmup call=1; measured calls=2,3 in [12,14].
    report = run_benchmark(
        adapter, ["private prompt"], repeats=2, warmup=1, clock=lambda: next(ticks)
    )
    assert adapter.calls == 3
    assert report["metrics"]["output_tokens_per_s"] == 1.0
    assert report["metrics"]["completed_requests"] == 2
    assert report["workload"]["concurrency"] == 1
    assert report["workload"]["warmup_requests"] == 1
    assert "private prompt" not in json.dumps(report)


def test_empty_prompt_and_invalid_repeats_rejected():
    with pytest.raises(ValueError):
        run_benchmark(None, [], repeats=1, warmup=0)
    with pytest.raises(ValueError):
        run_benchmark(None, ["x"], repeats=0, warmup=0)


def test_token_budget_rejected_before_generation():
    adapter = SimpleNamespace(encode=lambda _: [1, 2, 3])
    with pytest.raises(ValueError, match="max_model_len"):
        run_benchmark(adapter, ["x"], max_total_tokens=4, max_new_tokens=2)


def test_vllm_observes_cumulative_deltas_not_recounted_tokens():
    class FakeEngine:
        steps = 0

        def add_request(self, request_id, prompt, params):
            assert prompt == {"prompt_token_ids": [7, 8]}

        def has_unfinished_requests(self):
            return self.steps < 2

        def step(self):
            self.steps += 1
            ids = [3] if self.steps == 1 else [3, 4, 5]
            return [
                SimpleNamespace(
                    request_id="1",
                    finished=self.steps == 2,
                    outputs=[SimpleNamespace(token_ids=ids)],
                )
            ]

    adapter = object.__new__(VLLMAdapter)
    adapter.engine = FakeEngine()
    adapter._next_id = 0
    adapter.sampling_params = object()
    trace = adapter.generate([7, 8])
    assert len(trace.token_times_s) == 3
    assert trace.token_times_s[1] == trace.token_times_s[2]  # Coalesced engine observation.
    assert trace.ttft_s >= 0


def test_hf_streamer_skips_prompt_and_counts_token_ids(tmp_path):
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
