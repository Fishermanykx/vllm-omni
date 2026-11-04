# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""NPU adaptation of the shared VAE epilogue's rounding contract.

Keep the original FP32 ATen L2 reduction and native SiLU. The channel-first
persistent epilogue folds divide/cast/scale/gamma/zero-bias into one pass.
Ascend's div_rn alone differed at BF16 ties; correct its quotient using an
FMA residual. Measured parity is not a proof for every FP32 input/toolkit.
See diffusion/layers/vae_norm_epilogue.py for the shared CUDA/XPU precedent.
The explicit full backend also fuses the channel reduction and SiLU; its changed
reduction order is not bit-exact and requires separate quality acceptance.
"""

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton


@triton.jit
def _corrected_div(x, denominator):
    quotient = tl.div_rn(x, denominator)
    residual = tl.fma(-quotient, denominator, x)
    corrected = quotient + tl.div_rn(residual, denominator)
    return tl.where(tl.abs(quotient) == float("inf"), quotient, corrected)


@triton.jit
def _epilogue(
    X, DEN, GAMMA, Y, C: tl.constexpr, SP: tl.constexpr, SCALE: tl.constexpr, WORK: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    blocks: tl.constexpr = triton.cdiv(SP, BLOCK)
    for work in tl.range(tl.program_id(0), WORK, tl.num_programs(0)):
        n = work // (C * blocks)
        channel = work // blocks % C
        spatial = work % blocks * BLOCK + offsets
        valid = spatial < SP
        position = (n * C + channel).to(tl.int64) * SP + spatial
        den_position = n.to(tl.int64) * SP + spatial
        x = tl.load(X + position, valid, 0).to(tl.float32)
        denominator = tl.load(DEN + den_position, valid, 1)
        gamma = tl.load(GAMMA + channel).to(tl.float32)
        z = _corrected_div(x, denominator).to(tl.bfloat16).to(tl.float32)
        z = (z * SCALE).to(tl.bfloat16).to(tl.float32)
        z = (z * gamma).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + position, (z + 0.0).to(tl.bfloat16), valid)


@triton.jit
def _full_norm_silu(
    X, GAMMA, Y, C: tl.constexpr, SP: tl.constexpr, SCALE: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr
):
    n = tl.program_id(1)
    channel = tl.arange(0, BC)
    spatial = tl.program_id(0) * BS + tl.arange(0, BS)
    valid = (channel[:, None] < C) & (spatial[None, :] < SP)
    position = (n * C + channel[:, None]).to(tl.int64) * SP + spatial[None, :]
    x = tl.load(X + position, valid, 0).to(tl.float32)
    denominator = tl.maximum(tl.sqrt(tl.sum(x * x, axis=0)), 1e-12)
    gamma = tl.load(GAMMA + channel, channel < C, 0).to(tl.float32)
    z = _corrected_div(x, denominator[None, :]).to(tl.bfloat16).to(tl.float32)
    z = (z * SCALE).to(tl.bfloat16).to(tl.float32)
    z = (z * gamma[:, None]).to(tl.bfloat16).to(tl.float32) + 0.0
    result = z / (1.0 + tl.exp(-z))
    tl.store(Y + position, result.to(tl.bfloat16), valid)


def norm_silu(x, gamma, scale, *, backend="epilogue"):
    # The adapter declines unsupported layouts before entering this function.
    if backend not in ("epilogue", "full"):
        raise ValueError("Wan VAE norm backend must be epilogue or full")
    if backend == "full" and x.shape[1] in (96, 192, 384):
        n, c = x.shape[:2]
        spatial = x.numel() // (n * c)
        out = torch.empty_like(x)
        # SIMD C384/BS64 exceeded register capacity in screening; use SIMT.
        _full_norm_silu[(triton.cdiv(spatial, 64), n)](
            x,
            gamma.reshape(-1),
            out,
            c,
            spatial,
            scale,
            triton.next_power_of_2(c),
            64,
            enable_fp_fusion=False,
            force_simt_only=c >= 384,
        )
        return out
    denominator = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True).clamp_min_(1e-12)
    out = torch.empty_like(x)
    n, c = x.shape[:2]
    spatial = x.numel() // (n * c)
    work = n * c * triton.cdiv(spatial, 4096)
    _epilogue[(min(work, 80),)](
        x,
        denominator,
        gamma.reshape(-1),
        out,
        c,
        spatial,
        scale,
        work,
        4096,
        enable_fp_fusion=False,
    )
    return F.silu(out)
