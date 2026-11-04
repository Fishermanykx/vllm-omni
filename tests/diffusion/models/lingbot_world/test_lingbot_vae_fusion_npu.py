# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend950 regression coverage for the opt-in Wan VAE epilogue."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu, pytest.mark.A5]


def load_kernel():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_vae_norm_triton.py"
    spec = importlib.util.spec_from_file_location("_wan_fusion_npu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(1, 96, 1, 464, 832), (2, 192, 2, 7, 13), (1, 384, 1, 58, 104)])
def test_epilogue_matches_original_fp32_reduction(shape):
    module = load_kernel()
    with torch.inference_mode():
        torch.manual_seed(43)
        x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        gamma = torch.randn(shape[1], 1, 1, 1, device="npu", dtype=x.dtype)
        scale = math.sqrt(shape[1])
        expected = F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * scale * gamma + 0.0)
        actual = module.norm_silu(x, gamma, scale)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("value", [0.0, 1e-15, -1e-15, 1e-8])
@pytest.mark.parametrize("backend", ["epilogue", "full"])
def test_norm_preserves_l2_floor(value, backend):
    module = load_kernel()
    with torch.inference_mode():
        x = torch.full((1, 96, 1, 3, 5), value, device="npu", dtype=torch.bfloat16)
        gamma = torch.ones(96, 1, 1, 1, device="npu", dtype=x.dtype)
        expected = F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * math.sqrt(96) * gamma + 0.0)
        actual = module.norm_silu(x, gamma, math.sqrt(96), backend=backend)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [96, 192, 384])
def test_full_backend_channel_reduction_and_tail(channels):
    module = load_kernel()
    with torch.inference_mode():
        # Single nonzero channel has exact L2=1; exercises channel reduction,
        # both backend schedules, multi-batch indexing and partial spatial tiles.
        x = torch.zeros(2, channels, 1, 3, 5, device="npu", dtype=torch.bfloat16)
        x[:, channels - 1] = 1
        gamma = torch.ones(channels, 1, 1, 1, device="npu", dtype=x.dtype)
        scale = math.sqrt(channels)
        expected = F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * scale * gamma + 0.0)
        actual = module.norm_silu(x, gamma, scale, backend="full")
        assert torch.equal(actual, expected)
