# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Boundary/fallback contracts for native Wan norm; NPU accuracy is separate."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def load():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_vae_native_norm.py"
    spec = importlib.util.spec_from_file_location("_native_wan_norm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [0.0, 1e-15, 1e-8, 1.0])
@pytest.mark.parametrize("rank", [4, 5])
def test_l2_floor_and_bf16_affine_order(value, rank):
    x = torch.full((1, 4, *([2] * (rank - 2))), value, dtype=torch.bfloat16)
    gamma = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=x.dtype).reshape(4, *([1] * (rank - 2)))
    reference = F.normalize(x.float(), dim=1).to(x.dtype) * 2 * gamma + 0.0
    assert torch.equal(load().native_channel_norm(x, gamma, 2), reference)


def test_install_preserves_state_dict_and_cpu_fallback():
    class WanRMS_norm(torch.nn.Module):
        __module__ = "diffusers.models.test"

        def __init__(self):
            super().__init__()
            self.gamma = torch.nn.Parameter(torch.ones(4, 1, 1, 1))
            self.bias, self.channel_first, self.scale = 0.0, True, 2.0

        def forward(self, x):
            return x + 7

    norm = WanRMS_norm()
    vae = SimpleNamespace(decoder=torch.nn.Sequential(norm))
    state = vae.decoder.state_dict()
    module = load()
    assert module.install_native_norm(vae) == 1
    assert list(vae.decoder.state_dict()) == list(state)
    assert torch.equal(norm(torch.zeros(1)), torch.tensor([7.0]))
    with pytest.raises(ValueError, match="other fusion"):
        module.install_native_norm(vae)


def test_hook_is_rejected_before_any_patch():
    class WanRMS_norm(torch.nn.Identity):
        __module__ = "diffusers.models.test"
        channel_first, bias = True, 0.0

    norm = WanRMS_norm()
    norm.register_forward_hook(lambda *_: None)
    with pytest.raises(ValueError, match="hooks"):
        load().install_native_norm(SimpleNamespace(decoder=torch.nn.Sequential(norm)))
    assert not hasattr(norm, "_lingbot_native_norm_forward")
