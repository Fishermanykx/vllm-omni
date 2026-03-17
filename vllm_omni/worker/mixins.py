from __future__ import annotations

import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        self._init_omni_profiler()

    def _init_omni_profiler(self) -> None:
        vllm_config = getattr(self, "vllm_config", None)
        profiler_config = getattr(vllm_config, "profiler_config", None)
        if profiler_config is None or getattr(profiler_config, "profiler", None) != "torch":
            return

        from vllm_omni.profiler import OmniTorchProfilerWrapper, create_omni_profiler

        if isinstance(getattr(self, "profiler", None), OmniTorchProfilerWrapper):
            return

        worker_name = f"stage-rank-{getattr(self, 'rank', 0)}"
        self.profiler = create_omni_profiler(
            profiler_config=profiler_config,
            worker_name=worker_name,
            local_rank=getattr(self, "local_rank", 0),
        )
        logger.info("Replaced worker profiler with platform-specific Omni profiler for %s", worker_name)

    def profile(self, is_start: bool = True):
        from vllm_omni.profiler import OmniTorchProfilerWrapper

        profiler = getattr(self, "profiler", None)
        if not isinstance(profiler, OmniTorchProfilerWrapper):
            return super().profile(is_start)

        if is_start:
            profiler.set_trace_filename(f"stage_llm_{int(time.time())}")
            profiler.start()
            return None

        profiler.stop()
        return None
