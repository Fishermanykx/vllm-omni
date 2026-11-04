# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Wan channel L2 norm using the native FP32-accumulating reduction.

Read BF16 directly in the reduction; division still needs an FP32 input cast.
Keep the original L2 clamp,
BF16 normalization boundary, scalar scale, gamma and bias order. Reduction
implementations can round differently; this is not a bit-exactness guarantee.
"""

from types import MethodType

import torch


def native_channel_norm(x, gamma, scale, bias=0.0):
    denominator = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-12)
    normalized = (x.float() / denominator).to(x.dtype)
    return normalized * scale * gamma + bias


def _forward(self, x):
    if (
        x.device.type not in ("npu", "privateuseone")
        or torch.is_grad_enabled()
        or x.dtype != torch.bfloat16
        or self.gamma.dtype != x.dtype
        or x.ndim not in (4, 5)
    ):
        return self._lingbot_native_norm_forward(x)
    return native_channel_norm(x, self.gamma, self.scale, self.bias)


def install_native_norm(vae):
    """Patch only original decoder Wan norms; keep checkpoint names/SiLU intact."""
    targets = []
    for norm in vae.decoder.modules():
        if type(norm).__name__ != "WanRMS_norm" or not type(norm).__module__.startswith("diffusers."):
            continue
        if (
            not norm.channel_first
            or not isinstance(norm.bias, float)
            or norm.bias != 0.0
            or norm._forward_hooks
            or norm._forward_pre_hooks
            or hasattr(norm, "_lingbot_native_norm_forward")
            or hasattr(norm, "_lingbot_norm_forward")
        ):
            raise ValueError("Native Wan norm requires original channel-first norms without hooks/bias/other fusion")
        targets.append(norm)
    if not targets:
        raise ValueError("No original Wan decoder norms found")
    for norm in targets:
        norm._lingbot_native_norm_forward = norm.forward
        norm.forward = MethodType(_forward, norm)
    return len(targets)
