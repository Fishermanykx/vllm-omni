# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel T5 encoder model."""

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import (
    T5EncoderModel,
    attach_t5_encoder_hsdp_shard_conditions,
)

__all__ = ["T5EncoderModel", "attach_t5_encoder_hsdp_shard_conditions"]
