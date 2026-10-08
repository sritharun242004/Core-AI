# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false
import json
from pathlib import Path

import pytest
from fsdp_ring_lab.integration import check_launch, load_zero_config


@pytest.mark.parametrize("stage", [2, 3])
def test_zero_configs_have_consistent_fp32_batch_contract(stage):
    path = Path(__file__).parents[1] / "configs" / f"zero-stage{stage}.json"
    config = load_zero_config(path, batch_size=4)
    assert config["zero_optimization"]["stage"] == stage
    assert not config["bf16"]["enabled"]
    assert config["gradient_accumulation_steps"] == 1
    assert "offload_optimizer" not in config["zero_optimization"]


@pytest.mark.parametrize(
    "env,cuda,nccl,devices",
    [
        ({}, False, False, 0),
        ({"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}, True, True, 1),
        ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "2"}, True, True, 2),
        ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, True, False, 2),
        ({"WORLD_SIZE": "bad"}, True, True, 2),
    ],
)
def test_launch_fails_before_initializing_process_group(env, cuda, nccl, devices):
    with pytest.raises(RuntimeError):
        check_launch(env, cuda_available=cuda, nccl_available=nccl, device_count=devices)


def test_valid_launch_contract():
    assert check_launch(
        {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
        cuda_available=True,
        nccl_available=True,
        device_count=2,
    ) == (1, 1, 2)


def test_zero_config_rejects_incompatible_accumulation(tmp_path):
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 3},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_zero_config(path, batch_size=4)
