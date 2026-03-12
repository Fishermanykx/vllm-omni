# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def _decomposition_fft(x: torch.Tensor, cutoff_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split hidden states into low/high frequency components on sequence axis."""
    orig_dtype = x.dtype
    device = x.device

    x_fp32 = x.to(torch.float32)
    _, seq_len, _ = x_fp32.shape
    freq = torch.fft.fft(x_fp32, dim=1)

    freqs = torch.fft.fftfreq(seq_len, d=1.0, device=device)
    cutoff = cutoff_ratio * freqs.abs().max()

    low_mask = (freqs.abs() <= cutoff)[None, :, None]
    high_mask = ~low_mask

    low = torch.fft.ifft(freq * low_mask, dim=1).real.to(device=device, dtype=orig_dtype)
    high = torch.fft.ifft(freq * high_mask, dim=1).real.to(device=device, dtype=orig_dtype)
    return low, high


class _CacheWithFreqsContainer:
    def __init__(self, max_order: int):
        self.max_order = max_order
        self.derivative_low: list[torch.Tensor | None] = [None] * (max_order + 1)
        self.derivative_high: list[torch.Tensor | None] = [None] * (max_order + 1)
        self.temp_low: list[torch.Tensor | None] = [None] * (max_order + 1)
        self.temp_high: list[torch.Tensor | None] = [None] * (max_order + 1)

    def clear_derivatives(self) -> None:
        for i in range(self.max_order + 1):
            self.derivative_low[i] = None
            self.derivative_high[i] = None
            self.temp_low[i] = None
            self.temp_high[i] = None

    def _move_temp_to_derivative(self) -> None:
        for i in range(self.max_order + 1):
            if self.temp_low[i] is not None:
                self.derivative_low[i] = self.temp_low[i]
            if self.temp_high[i] is not None:
                self.derivative_high[i] = self.temp_high[i]
        for i in range(self.max_order + 1):
            self.temp_low[i] = None
            self.temp_high[i] = None

    def derivatives_computation(
        self,
        hidden_states: torch.Tensor,
        distance: int,
        low_freqs_order: int,
        high_freqs_order: int,
    ) -> None:
        x_low, x_high = _decomposition_fft(hidden_states, cutoff_ratio=0.1)
        self.temp_low[0] = x_low
        self.temp_high[0] = x_high

        safe_distance = max(int(distance), 1)

        for i in range(min(low_freqs_order, self.max_order)):
            if self.derivative_low[i] is None or self.temp_low[i] is None:
                break
            self.temp_low[i + 1] = (self.temp_low[i] - self.derivative_low[i]) / safe_distance

        for i in range(min(high_freqs_order, self.max_order)):
            if self.derivative_high[i] is None or self.temp_high[i] is None:
                break
            self.temp_high[i + 1] = (self.temp_high[i] - self.derivative_high[i]) / safe_distance

        self._move_temp_to_derivative()

    def taylor_formula(self, distance: int) -> torch.Tensor:
        low_out = 0
        high_out = 0
        for i, deriv in enumerate(self.derivative_low):
            if deriv is None:
                break
            low_out = low_out + (deriv * (distance**i)) / math.factorial(i)
        for i, deriv in enumerate(self.derivative_high):
            if deriv is None:
                break
            high_out = high_out + (deriv * (distance**i)) / math.factorial(i)
        return low_out + high_out


@dataclass
class TaylorCacheRuntimeConfig:
    interval: int = 4
    order: int = 2
    enable_first_enhance: bool = False
    first_enhance_steps: int = 3
    enable_tailing_enhance: bool = False
    tailing_enhance_steps: int = 1
    low_freqs_order: int = 0
    high_freqs_order: int = 2


class HunyuanTaylorCacheManager:
    """Runtime state machine for HunyuanImage3 Taylor Cache."""

    def __init__(self, config: TaylorCacheRuntimeConfig):
        self.config = config
        max_order = max(config.order, config.low_freqs_order, config.high_freqs_order)
        self.cache = _CacheWithFreqsContainer(max_order=max_order)
        self.num_steps = 0
        self.current_step = 0
        self.counter = 0
        self.last_full_computation_step = 0
        self.last_past_key_values = None

    def reset(self, num_steps: int) -> None:
        self.num_steps = int(num_steps)
        self.current_step = 0
        self.counter = 0
        self.last_full_computation_step = 0
        self.last_past_key_values = None
        self.cache.clear_derivatives()

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def should_full_compute(self) -> bool:
        if self.current_step == 0:
            return True
        if self.counter == max(self.config.interval, 1) - 1:
            return True
        if self.config.enable_first_enhance and self.current_step < self.config.first_enhance_steps:
            return True
        if self.config.enable_tailing_enhance and self.current_step >= self.num_steps - self.config.tailing_enhance_steps:
            return True
        return False

    def update_from_full(self, hidden_states: torch.Tensor, past_key_values: Any) -> None:
        self.counter = 0
        if not (self.config.enable_first_enhance and self.current_step < self.config.first_enhance_steps - 1):
            distance = self.current_step - self.last_full_computation_step
            self.cache.derivatives_computation(
                hidden_states,
                distance=distance,
                low_freqs_order=self.config.low_freqs_order,
                high_freqs_order=self.config.high_freqs_order,
            )
        self.last_full_computation_step = self.current_step
        self.last_past_key_values = past_key_values

    def forecast(self) -> tuple[torch.Tensor, Any]:
        self.counter += 1
        hidden_states = self.cache.taylor_formula(distance=self.counter)
        return hidden_states, self.last_past_key_values

    def maybe_finalize(self) -> None:
        if self.current_step == self.num_steps - 1:
            self.cache.clear_derivatives()
            self.last_past_key_values = None
            self.counter = 0
            self.last_full_computation_step = 0


def enable_taylor_cache_for_hunyuan_image3(
    pipeline: Any,
    runtime_config: TaylorCacheRuntimeConfig,
) -> HunyuanTaylorCacheManager:
    """Enable Taylor Cache for HunyuanImage3Pipeline."""
    manager = HunyuanTaylorCacheManager(runtime_config)
    # HunyuanImage3 runtime reads cache manager from the outer pipeline object.
    setattr(pipeline, "_taylor_cache_manager", manager)
    # Keep a mirrored reference on nested model for compatibility with older call sites.
    if hasattr(pipeline, "model"):
        setattr(pipeline.model, "_taylor_cache_manager", manager)
    return manager


CUSTOM_TAYLOR_ENABLERS: dict[str, Callable[[Any, TaylorCacheRuntimeConfig], HunyuanTaylorCacheManager]] = {
    "HunyuanImage3Pipeline": enable_taylor_cache_for_hunyuan_image3,
}


class TaylorCacheBackend(CacheBackend):
    """Taylor Cache backend for HunyuanImage3 diffusion pipeline."""

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._manager: HunyuanTaylorCacheManager | None = None

    def _build_runtime_config(self) -> TaylorCacheRuntimeConfig:
        return TaylorCacheRuntimeConfig(
            interval=max(int(self.config.taylor_cache_interval), 1),
            order=max(int(self.config.taylor_cache_order), 0),
            enable_first_enhance=bool(self.config.taylor_cache_enable_first_enhance),
            first_enhance_steps=max(int(self.config.taylor_cache_first_enhance_steps), 0),
            enable_tailing_enhance=bool(self.config.taylor_cache_enable_tailing_enhance),
            tailing_enhance_steps=max(int(self.config.taylor_cache_tailing_enhance_steps), 0),
            low_freqs_order=max(int(self.config.taylor_cache_low_freqs_order), 0),
            high_freqs_order=max(int(self.config.taylor_cache_high_freqs_order), 0),
        )

    def enable(self, pipeline: Any) -> None:
        pipeline_name = pipeline.__class__.__name__
        enabler = CUSTOM_TAYLOR_ENABLERS.get(pipeline_name)

        if enabler is None:
            logger.warning(
                "TaylorCacheBackend currently only supports %s, got %s",
                ", ".join(sorted(CUSTOM_TAYLOR_ENABLERS)),
                pipeline_name,
            )
            self.enabled = False
            self._manager = None
            return

        self._manager = enabler(pipeline, self._build_runtime_config())
        self.enabled = True
        logger.info(
            "Taylor Cache enabled on %s: interval=%d order=%d first_enhance=%s tailing_enhance=%s",
            pipeline_name,
            self._manager.config.interval,
            self._manager.config.order,
            self._manager.config.enable_first_enhance,
            self._manager.config.enable_tailing_enhance,
        )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        if not self.enabled or self._manager is None:
            return
        self._manager.reset(num_steps=num_inference_steps)
        if verbose:
            logger.debug("Taylor Cache state refreshed (num_inference_steps=%d)", num_inference_steps)
