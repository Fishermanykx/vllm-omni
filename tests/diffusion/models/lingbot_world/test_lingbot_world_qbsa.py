# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the Ada-BSA-estimator -> Eagle QBSA adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _load():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_qbsa.py"
    spec = importlib.util.spec_from_file_location("_lingbot_qbsa_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, False, "0.85", float("nan"), float("inf")])
def test_invalid_cdf_rejected(value):
    with pytest.raises(ValueError, match="finite number"):
        _load().validate_cdf(value)


def test_cdf_disabled_and_selected():
    module = _load()
    assert module.validate_cdf(None) == 0
    assert module.validate_cdf(0.85) == 0.85
    assert module.validate_cdf(1) == 1


@pytest.mark.parametrize("q_len,k_len", [(128, 128), (257, 2049), (4524, 27144)])
def test_native_estimator_contract_and_padding_crop(monkeypatch, q_len, k_len):
    module = _load()
    ops = ModuleType("mindiesd.layers._custom_ops")
    seen = []
    nq, nk = (q_len + 127) // 128, (k_len + 127) // 128
    mask = torch.zeros(1, 2, nq, ((nk + 31) // 32) * 32, dtype=torch.int8)
    mask[..., 0] = 1
    mask[..., nk - 1] = 1

    def estimate(**kwargs):
        seen.append(kwargs)
        return mask, mask[..., :nk].sum(-1).to(torch.int32)

    setattr(ops, "sparse_block_estimate", estimate)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    q, k = torch.randn(1, 2, q_len, 128), torch.randn(1, 2, k_len, 128)
    result = module.cdf_block_mask(q, k, cdf=0.85, scale=128**-0.5)
    assert result.shape == (1, 2, nq, nk)
    assert result.is_contiguous() and result.dtype == torch.int8
    torch.testing.assert_close(result, mask[..., :nk])
    kw = seen[0]
    assert kw["threshold"] == 0.85 and kw["row_sparse"] == 1.0
    assert kw["stride"] == 8 and kw["scale_value"] == 128**-0.5 / 8
    assert kw["actual_seq_lengths"] is None
    assert kw["actual_seq_lengths_kv"] is None
    assert kw["causal"] is False and kw["keep_sink"] and kw["keep_recent"]


def test_eqbsa_rectangular_tails_and_quantization_abi(monkeypatch):
    module = _load()
    ops = ModuleType("mindiesd.layers._custom_ops")
    quant = ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3")
    provider = ModuleType("torch_npu")
    setattr(provider, "float8_e4m3fn", torch.float8_e4m3fn)
    calls = []

    def estimate(**kwargs):
        return torch.ones(1, 2, 3, 32, dtype=torch.int8), torch.full((1, 2, 3), 17, dtype=torch.int32)

    def quantize(q, k, v, *, block_size_q, layout):
        assert block_size_q == 64 and layout == "BNSD"
        assert q.shape == (1, 2, 257, 128) and k.shape == v.shape == (1, 2, 2049, 128)
        return (
            q.to(torch.int8),
            k.to(torch.int8),
            v.to(torch.float8_e4m3fn),
            torch.ones(1, 2, 5),
            torch.ones(1, 2, 33),
            torch.ones(1, 2, 128),
        )

    def eagle(**kwargs):
        calls.append(kwargs)
        assert kwargs["block_sparse_mask"].shape == (1, 2, 3, 17)
        assert kwargs["value"].dtype == torch.int8
        assert kwargs["value_dtype"] == torch.float8_e4m3fn
        assert kwargs["actual_seq_lengths"] == [257]
        assert kwargs["actual_seq_lengths_kv"] == [2049]
        assert kwargs["inner_precise"] == 4
        assert kwargs["block_shape"] == [128, 128]
        return kwargs["query"].to(torch.bfloat16), None

    setattr(ops, "sparse_block_estimate", estimate)
    setattr(ops, "eagle_quant_block_sparse_attention", eagle)
    setattr(quant, "_eagle_qbsa_quant_qkv", quantize)
    for obj in (ops, quant, provider):
        monkeypatch.setitem(sys.modules, obj.__name__, obj)
    q = torch.randn(257, 2, 128, dtype=torch.bfloat16)
    k = torch.randn(2049, 2, 128, dtype=torch.bfloat16)
    out = module.eagle_qbsa_attention(q, k, k, cdf=0.85, scale=128**-0.5)
    assert len(calls) == 1 and out.shape == q.shape and out.is_contiguous()
    assert out.dtype == torch.bfloat16


def test_eqbsa_rejects_unsupported_dtype_before_provider_import():
    q = torch.zeros(3, 2, 128)
    with pytest.raises(ValueError, match="BF16"):
        _load().eagle_qbsa_attention(q, q, q, cdf=0.85, scale=0.1)
