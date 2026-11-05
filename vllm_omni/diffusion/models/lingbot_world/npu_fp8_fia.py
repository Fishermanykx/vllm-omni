# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense FP8 EagleFIA using MindIE-SD's coupled C8V16/V512 mode."""

import math

import torch


def eagle_fp8_fia(query, key, value, *, scale):
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("EagleFIA requires TND Q and matching TND K/V")
    if query.shape[1:] != key.shape[1:] or query.shape[-1] != 128:
        raise ValueError("LingBot EagleFIA requires matching heads and head_dim128")
    if min(query.shape[0], key.shape[0]) <= 0:
        raise ValueError("EagleFIA requires nonempty Q/K")
    if any(x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("LingBot EagleFIA requires BF16 inputs")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("EagleFIA scale must be positive and finite")
    from mindiesd.layers.flash_attn.quant_flash_attn import quant_attention
    from mindiesd.quantization.mode import FP8FAMode

    q, k, v = (x.permute(1, 0, 2).unsqueeze(0).contiguous() for x in (query, key, value))
    out = quant_attention(
        q,
        k,
        v,
        precision="fp8",
        layout="BNSD",
        scale=scale,
        fp8_fa_mode=FP8FAMode.C8V16_TILING512,
    )
    if out.shape != q.shape or out.dtype != query.dtype:
        raise RuntimeError("MindIE FP8 EagleFIA output contract mismatch")
    return out[0].permute(1, 0, 2).contiguous()
