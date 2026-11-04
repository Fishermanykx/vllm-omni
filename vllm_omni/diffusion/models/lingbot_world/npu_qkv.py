# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan-specific QKV preprocessing; provider imported only when requested."""

import torch


def split_qkv_reference(qkv, q_weight, k_weight, cos, sin, *, eps=1e-6):
    """CPU/device oracle retaining across-head RMS and the BF16 boundary."""
    q, k, v = qkv.split(q_weight.numel(), dim=-1)

    def apply(x, weight):
        xf = x.float()
        norm = (xf / torch.sqrt(xf.square().sum(-1, keepdim=True) / x.shape[-1] + eps) * weight.float()).to(x.dtype)
        pairs = norm.unflatten(-1, (-1, 128)).unflatten(-1, (64, 2))
        a, b = pairs.unbind(-1)
        c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).flatten(-2).to(x.dtype)

    return apply(q, q_weight), apply(k, k_weight), v


def split_qkv_wan_npu(qkv, q_weight, k_weight, cos, sin, *, eps):
    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[-1] != 15360 or qkv.dtype != torch.bfloat16:
        raise ValueError("Wan QKV fusion requires BF16 [1, tokens, 15360] at TP1")
    tokens = qkv.shape[1]
    if q_weight.numel() != 5120 or k_weight.numel() != 5120 or cos.numel() != tokens * 64 or sin.shape != cos.shape:
        raise ValueError("Wan QKV fusion requires 5120-wide Q/K weights and per-token 64-pair RoPE")
    from vllm_omni.diffusion.models.lingbot_world.npu_qkv_triton import launch

    q, k = launch(qkv.contiguous(), q_weight, k_weight, cos.reshape(tokens, 64), sin.reshape(tokens, 64), eps)
    return q.unsqueeze(0), k.unsqueeze(0), qkv[..., 10240:]
