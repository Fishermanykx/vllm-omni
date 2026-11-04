# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).parents[4] / "vllm_omni/diffusion/models/lingbot_world"


def load(name):
    spec = importlib.util.spec_from_file_location("_lingbot_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_mxfp8_selects_only_attention_and_ffn(monkeypatch):
    module = load("npu_quantization")
    provider = ModuleType("vllm_omni.quantization.mxfp8_config")
    seen = []
    provider.DiffusionMXFP8Config = lambda **kw: seen.append(kw) or kw
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    result = module.online_mxfp8_config("mxfp8", None, num_layers=40)
    assert result["is_checkpoint_mxfp8_serialized"] is False
    ignored = result["ignored_layers"]
    assert len(ignored) == 82 and len(set(ignored)) == 82
    assert "transformer.blocks.39.cam_injector_layer2" in ignored
    assert all("self_attn" not in x and "cross_attn" not in x and ".ffn" not in x for x in ignored)
    existing = object()
    assert module.online_mxfp8_config(None, existing, num_layers=40) is existing
    with pytest.raises(ValueError, match="not both"):
        module.online_mxfp8_config("mxfp8", existing, num_layers=40)


@pytest.mark.parametrize("mode", ["int8", "fp8", True, ""])
def test_online_quant_rejects_non_mxfp8(mode):
    with pytest.raises(ValueError, match="mxfp8"):
        load("npu_quantization").online_mxfp8_config(mode, None, num_layers=40)


@pytest.mark.parametrize("tokens", [1, 7])
def test_wan_qkv_oracle_preserves_across_head_norm_and_adjacent_rope(tokens):
    module = load("npu_qkv")
    torch.manual_seed(13)
    x = torch.randn(1, tokens, 15360, dtype=torch.bfloat16)
    qw, kw = torch.randn(5120), torch.randn(5120)
    cos, sin = torch.randn(tokens, 64, dtype=torch.bfloat16), torch.randn(tokens, 64, dtype=torch.bfloat16)
    outputs = module.split_qkv_reference(x, qw, kw, cos, sin)
    for source, weight, actual in zip(x.split(5120, -1)[:2], (qw, kw), outputs[:2]):
        xf = source.float()
        norm = (xf / torch.sqrt(xf.square().sum(-1, keepdim=True) / 5120 + 1e-6) * weight).to(x.dtype)
        heads = norm.reshape(1, tokens, 40, 128)
        a, b = heads[..., ::2], heads[..., 1::2]
        expected = torch.stack(
            (a * cos[:, None] - b * sin[:, None], a * sin[:, None] + b * cos[:, None]), -1
        ).reshape_as(actual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(outputs[2], x[..., 10240:], atol=0, rtol=0)


def test_qkv_fusion_rejects_wrong_geometry_before_provider_import():
    with pytest.raises(ValueError, match="TP1"):
        load("npu_qkv").split_qkv_wan_npu(torch.zeros(1, 3, 100), None, None, None, None, eps=1e-6)


class WanLikeConv(torch.nn.Conv3d):
    def __init__(self):
        super().__init__(2, 3, 3, padding=0)
        self._padding = (1, 1, 1, 1, 2, 0)
        self._lingbot_padding_verdicts = {}
        self._lingbot_original_forward = self.forward

    def forward(self, x, cache_x=None):
        p = list(self._padding)
        if cache_x is not None:
            x = torch.cat((cache_x, x), 2)
            p[4] -= cache_x.shape[2]
        return super().forward(torch.nn.functional.pad(x, p))


@pytest.mark.parametrize("history", [0, 1, 2, 3])
def test_vae_padding_preserves_temporal_cache_and_crop(history):
    module = load("npu_vae")
    conv = WanLikeConv()
    x = torch.randn(1, 2, 3, 7, 9)
    cache = torch.randn(1, 2, history, 7, 9) if history else None
    expected = conv(x, cache)
    torch.testing.assert_close(module.spatial_padding_candidate(conv, x, cache), expected)
    torch.testing.assert_close(module.verified_spatial_forward(conv, x, cache), expected, atol=0, rtol=0)


def test_vae_padding_mismatch_returns_reference_and_remembers_fallback(monkeypatch):
    module = load("npu_vae")
    conv = WanLikeConv()
    x = torch.randn(1, 2, 3, 7, 9)
    calls = []

    def wrong(*args):
        calls.append(1)
        return conv(x) + 1

    monkeypatch.setattr(module, "spatial_padding_candidate", wrong)
    for _ in range(2):
        torch.testing.assert_close(module.verified_spatial_forward(conv, x), conv(x), atol=0, rtol=0)
    assert len(calls) == 1


def test_vae_norm_installer_matches_real_diffusers_structure_and_preserves_parameters():
    from diffusers import AutoencoderKLWan

    module = load("npu_vae_norm")
    with torch.device("meta"):
        vae = AutoencoderKLWan()
    names_before = set(vae.state_dict())
    assert module.install_norm_silu(vae) > 0
    assert set(vae.state_dict()) == names_before
    norm = next(m for m in vae.decoder.modules() if hasattr(m, "_lingbot_norm_forward"))
    fresh = type(norm)(dim=4, images=False).to(dtype=torch.bfloat16)
    x = torch.randn(1, 4, 2, 3, 5, dtype=torch.bfloat16)
    fresh._lingbot_norm_forward = fresh.forward
    expected = torch.nn.functional.silu(fresh(x))
    torch.testing.assert_close(module._norm_silu(fresh, x), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="already installed"):
        module.install_norm_silu(vae)
