# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Wan NCTHW L2 norm epilogue + SiLU; FP32 reduction stays in ATen.

Preserves the norm/scale/gamma BF16 boundaries. SiLU/reduction numerical
acceptance on Ascend is pending; this is not a bit-exactness claim.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _epilogue(X, DEN, GAMMA, Y, SIZE: tl.constexpr, C: tl.constexpr, SP: tl.constexpr, SCALE: tl.constexpr):
    i = tl.program_id(0) * 512 + tl.arange(0, 512)
    valid = i < SIZE
    channel = i // SP % C
    n = i // (SP * C)
    denominator = tl.load(DEN + n * SP + i % SP, valid, 1)
    x = tl.load(X + i, valid, 0).to(tl.float32)
    g = tl.load(GAMMA + channel, valid, 0).to(tl.float32)
    normalized = (x / denominator).to(tl.bfloat16).to(tl.float32)
    scaled = (normalized * SCALE).to(tl.bfloat16).to(tl.float32)
    value = (scaled * g).to(tl.bfloat16).to(tl.float32) + 0.0
    result = value / (1.0 + tl.exp(-value))
    tl.store(Y + i, result.to(tl.bfloat16), valid)


def norm_silu(x, gamma, scale):
    x = x.contiguous()
    denominator = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)
    out = torch.empty_like(x)
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    _epilogue[(triton.cdiv(x.numel(), 512),)](
        x,
        denominator.contiguous(),
        gamma.reshape(-1),
        out,
        x.numel(),
        x.shape[1],
        spatial,
        scale,
        enable_fp_fusion=False,
    )
    return out
