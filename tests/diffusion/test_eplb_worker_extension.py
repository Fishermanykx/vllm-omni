# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.worker.eplb_worker_extension import (
    get_eplb_config_dict,
    is_eplb_requested,
    uses_dynamic_eplb,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class DummyConfig:
    def __init__(self, eplb_config=None):
        self.eplb_config = eplb_config


def test_get_eplb_config_dict_defaults_to_empty():
    assert get_eplb_config_dict(DummyConfig()) == {}


def test_static_eplb_request_is_detected():
    cfg = DummyConfig(eplb_config={"expert_map_path": "/tmp/expert_map.json"})
    assert is_eplb_requested(cfg) is True
    assert uses_dynamic_eplb(cfg) is False


def test_dynamic_eplb_request_is_detected():
    cfg = DummyConfig(eplb_config={"dynamic_eplb": True, "num_redundant_experts": 4})
    assert is_eplb_requested(cfg) is True
    assert uses_dynamic_eplb(cfg) is True
