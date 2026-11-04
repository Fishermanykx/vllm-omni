# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit instance-local original Wan norm contract after the NPU Wan patch.

The shared Wan2.2 import replaces diffusers' norm constructor globally. Do not
silently treat its RMSNormVAE as the original L2 norm. This opt-in restores the
checkpoint's diffusers expression only on this VAE instance, before fusion.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def is_restored_original_norm(norm):
    return (
        type(norm).__name__ == "RMSNormVAE"
        and type(norm).__module__ == "vllm_omni.diffusion.models.wan2_2.norm"
        and getattr(norm, "_lingbot_original_wan_norm", False) is True
    )


def _forward(self, x):
    source = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    normalized = F.normalize(source, dim=1).to(x.dtype)
    return normalized * self.scale * self.gamma + (0.0 if self.bias is None else self.bias)


def install_original_wan_norm(vae):
    targets = []
    for norm in vae.modules():
        if type(norm).__name__ != "RMSNormVAE" or type(norm).__module__ != "vllm_omni.diffusion.models.wan2_2.norm":
            continue
        if (
            not norm.channel_first
            or norm.bias is not None
            or norm._forward_hooks
            or norm._forward_pre_hooks
            or hasattr(norm, "_lingbot_original_wan_norm")
        ):
            raise ValueError("Original Wan norm requires unmodified channel-first RMSNormVAE without bias/hooks")
        targets.append(norm)
    for norm in targets:
        norm._lingbot_framework_norm_forward = norm.forward
        norm._lingbot_original_wan_norm = True
        norm.forward = MethodType(_forward, norm)
    return len(targets)
