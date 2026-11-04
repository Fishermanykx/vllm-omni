# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit original-norm restoration must not reinterpret framework RMSNorm."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def load():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_vae_original_norm.py"
    spec = importlib.util.spec_from_file_location("_original_wan_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_norm():
    class RMSNormVAE(torch.nn.Module):
        __module__ = "vllm_omni.diffusion.models.wan2_2.norm"

        def __init__(self):
            super().__init__()
            self.gamma = torch.nn.Parameter(torch.ones(4, 1, 1, 1, dtype=torch.bfloat16))
            self.bias, self.channel_first, self.scale = None, True, 2.0

        def forward(self, x):
            return x + 123

    return RMSNormVAE()


@pytest.mark.parametrize("value", [0.0, 1e-15, 1.0])
def test_explicit_restore_is_instance_local_and_preserves_parameters(value):
    norm, untouched = make_norm(), make_norm()
    module = load()
    names = set(norm.state_dict())
    gamma = norm.gamma
    assert not module.is_restored_original_norm(norm)
    assert module.install_original_wan_norm(norm) == 1
    x = torch.full((1, 4, 1, 2, 3), value, dtype=torch.bfloat16)
    expected = F.normalize(x.float(), dim=1).to(x.dtype) * 2.0 * gamma + 0.0
    assert torch.equal(norm(x), expected)
    assert module.is_restored_original_norm(norm)
    assert not module.is_restored_original_norm(untouched)
    assert torch.equal(untouched(x), x + 123)
    assert norm.gamma is gamma and set(norm.state_dict()) == names
    with pytest.raises(ValueError, match="unmodified"):
        module.install_original_wan_norm(norm)
