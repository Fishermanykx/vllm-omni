# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Experimental TP1 across-head RMSNorm + adjacent-pair Wan RoPE.

Inspired by Ascend's split_qkv_tp_rmsnorm_rope; no per-head normalization or
Qwen MRoPE axis selection. NPU compilation/numerical acceptance is separate.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _kernel(X, QW, KW, C, S, Y, TOKENS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    group = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    valid = d < 5120
    x = tl.load(X + token * 15360 + group * 5120 + d, valid, 0).to(tl.float32)
    if group == 0:
        w = tl.load(QW + d, valid, 0).to(tl.float32)
    else:
        w = tl.load(KW + d, valid, 0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, 0) / 5120 + EPS)
    # Restore the materialized norm boundary before rotating adjacent pairs.
    n = (x / rms * w).to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(n, d ^ 1, axis=0)
    pair = d % 128 // 2
    c = tl.load(C + token * 64 + pair, valid, 0).to(tl.float32)
    s = tl.load(S + token * 64 + pair, valid, 0).to(tl.float32)
    out = n * c + tl.where(d % 2 == 0, -partner, partner) * s
    tl.store(Y + (group * TOKENS + token) * 5120 + d, out.to(tl.bfloat16), valid)


def launch(qkv, q_weight, k_weight, cos, sin, eps):
    tokens = qkv.shape[1]
    output = torch.empty((2, tokens, 5120), device=qkv.device, dtype=qkv.dtype)
    if tokens:
        _kernel[(tokens, 2)](
            qkv,
            q_weight,
            k_weight,
            cos.contiguous(),
            sin.contiguous(),
            output,
            tokens,
            eps,
            8192,
            enable_fp_fusion=False,
        )
    return output[0], output[1]
