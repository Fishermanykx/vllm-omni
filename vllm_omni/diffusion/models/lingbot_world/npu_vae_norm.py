# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in original Wan L2 norm + SiLU, retaining registered checkpoint names."""

from types import MethodType

import torch
import torch.nn.functional as F


def _norm_silu(self, x):
    if x.device.type not in ("npu", "privateuseone") or torch.is_grad_enabled():
        return F.silu(self._lingbot_norm_forward(x))
    if (
        x.dtype != torch.bfloat16
        or self.gamma.dtype != x.dtype
        or x.ndim not in (4, 5)
        or not x.is_contiguous()
        or x.numel() == 0
        or torch.compiler.is_compiling()
    ):
        return F.silu(self._lingbot_norm_forward(x))
    from vllm_omni.diffusion.models.lingbot_world.npu_vae_norm_triton import norm_silu

    return norm_silu(x, self.gamma, self.scale)


def _identity(self, x):
    return x


def install_norm_silu(vae):
    """Only original diffusers norms; never reinterpret RMSNormVAE's epsilon."""
    targets = []
    for block in vae.decoder.modules():
        name = type(block).__name__
        if not type(block).__module__.startswith("diffusers."):
            continue
        names = ("norm1", "norm2") if name == "WanResidualBlock" else ("norm_out",) if name == "WanDecoder3d" else ()
        if not names:
            continue
        activation = getattr(block, "nonlinearity", None)
        if type(activation) is not torch.nn.SiLU or activation.inplace:
            raise ValueError("Wan norm fusion requires plain out-of-place SiLU")
        norms = [getattr(block, item) for item in names]
        for norm in norms:
            if (
                type(norm).__name__ != "WanRMS_norm"
                or not norm.channel_first
                or not isinstance(norm.bias, float)
                or norm.bias != 0.0
                or norm._forward_hooks
                or norm._forward_pre_hooks
                or hasattr(norm, "_lingbot_native_norm_forward")
            ):
                raise ValueError("Wan norm fusion requires original channel-first WanRMS_norm without hooks/bias")
            if hasattr(norm, "_lingbot_norm_forward"):
                raise ValueError("Wan norm fusion already installed")
        if activation._forward_hooks or activation._forward_pre_hooks:
            raise ValueError("Wan norm fusion does not bypass activation hooks")
        targets.append((norms, activation))
    if not targets:
        raise ValueError("No supported Wan decoder norm/SiLU pairs found")
    for norms, activation in targets:
        for norm in norms:
            norm._lingbot_norm_forward = norm.forward
            norm.forward = MethodType(_norm_silu, norm)
        activation.forward = MethodType(_identity, activation)
    return sum(len(norms) for norms, _ in targets)
