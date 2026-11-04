# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental rectangular Eagle QBSA with the Ada-BSA CDF estimator.

Use the native ascend950 SparseBlockEstimate operator directly; the older
high-level ada_bsa wrapper still rejects A5 devices.
CDF refers to estimated block probability mass, never a fixed keep ratio.
"""

import math

import torch

BLOCK_SIZE = 128


def validate_cdf(cdf: float | None) -> float:
    if cdf is None:
        return 0.0
    if isinstance(cdf, bool) or not isinstance(cdf, (float, int)) or not math.isfinite(cdf) or not 0 < cdf <= 1:
        raise ValueError("lingbot_npu_qbsa_cdf must be a finite number in (0, 1], or null to disable")
    return float(cdf)


def cdf_block_mask(query: torch.Tensor, key: torch.Tensor, *, cdf: float, scale: float) -> torch.Tensor:
    """Reuse Ada-BSA native KV selection, trimming aligned padding for EQBSA.

    The native estimator supports different Q/K lengths and partial blocks.
    Recompute on every forward; no stale mask reuse across diffusion steps.
    Scale/stride matches MindIE-SD get_estimate_mask, with no fixed keep-ratio
    override. keep_sink/keep_recent protect boundary blocks (not full frames).
    """
    validate_cdf(cdf)
    if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
        raise ValueError("QBSA mask requires matching BNSD batch/head dimensions")
    if query.shape[-1] != key.shape[-1] or min(query.shape[-2], key.shape[-2]) <= 0:
        raise ValueError("QBSA mask requires matching head dimensions and nonempty sequences")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("QBSA attention scale must be finite and positive")
    from mindiesd.layers._custom_ops import sparse_block_estimate

    stride = 8
    mask, _counts = sparse_block_estimate(
        query=query,
        key=key,
        # A5 BNSD tiling requires absent length arrays. These tensors already
        # contain exactly the valid tokens; their shapes encode both lengths.
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        input_layout="BNSD",
        stride=stride,
        sparse_size=BLOCK_SIZE,
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        scale_value=scale / stride,
        threshold=cdf,
        causal=False,
        keep_sink=True,
        keep_recent=True,
        row_sparse=1.0,
    )
    nq = (query.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE
    nk = (key.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE
    expected = (*query.shape[:2], nq, ((nk + 31) // 32) * 32)
    if tuple(mask.shape) != expected or mask.dtype != torch.int8:
        raise RuntimeError("MindIE SparseBlockEstimate mask contract mismatch")
    # Ada-BSA pads KV mask rows to 32 columns; EQBSA accepts valid block columns.
    return mask[..., :nk].contiguous()


def eagle_qbsa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, cdf: float, scale: float):
    """TND singleton self attention -> mixed INT8 Q/K, FP8 V -> TND BF16.

    Uses the version-matched Ada-BSA estimator, MindIE quantization helper and native EQBSA operator,
    rather than rf_v3's equal-length spatial-rearrangement entry point.
    Requested providers fail explicitly if unavailable; no dense disguise.
    """
    validate_cdf(cdf)
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("LingBot QBSA requires TND Q and matching TND K/V")
    if query.shape[1:] != key.shape[1:] or query.shape[-1] != 128:
        raise ValueError("LingBot QBSA requires matching heads and head_dim=128")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise ValueError("LingBot QBSA requires BF16 Q/K/V")
    import torch_npu
    from mindiesd.layers._custom_ops import eagle_quant_block_sparse_attention
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import _eagle_qbsa_quant_qkv

    q, k, v = (t.permute(1, 0, 2).unsqueeze(0).contiguous() for t in (query, key, value))
    mask = cdf_block_mask(q, k, cdf=cdf, scale=scale)
    qq, kq, vq, qs, ks, vs = _eagle_qbsa_quant_qkv(q, k, v, block_size_q=64, layout="BNSD")
    output, _ = eagle_quant_block_sparse_attention(
        query=qq,
        key=kq,
        value=vq.view(torch.int8),
        block_sparse_mask=mask,
        block_shape=[BLOCK_SIZE, BLOCK_SIZE],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=key.shape[1],
        scale_value=scale,
        inner_precise=4,
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[key.shape[0]],
        softmax_lse_flag=0,
        query_scale=qs,
        key_scale=ks,
        value_scale=vs,
        query_dtype=torch.int8,
        key_dtype=torch.int8,
        value_dtype=torch_npu.float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )
    return output[0].permute(1, 0, 2).contiguous()
