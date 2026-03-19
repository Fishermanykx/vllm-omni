# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from multiprocessing import Manager
from typing import Any

from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def get_eplb_config_dict(od_config: Any) -> dict[str, Any]:
    raw_cfg = getattr(od_config, "eplb_config", None)
    if isinstance(raw_cfg, Mapping):
        return dict(raw_cfg)
    return {}


def is_eplb_requested(od_config: Any) -> bool:
    cfg = get_eplb_config_dict(od_config)
    return bool(cfg.get("dynamic_eplb") or cfg.get("expert_map_path") or cfg.get("expert_map_record_path"))


def uses_dynamic_eplb(od_config: Any) -> bool:
    cfg = get_eplb_config_dict(od_config)
    return bool(cfg.get("dynamic_eplb") or cfg.get("expert_map_record_path"))


class EplbWorkerExtension:
    def _is_npu_eplb_target(self) -> bool:
        return current_omni_platform.device_type == "npu" and is_eplb_requested(self.od_config)

    def _ensure_vllm_additional_config(self) -> None:
        if self.vllm_config.additional_config is None:
            self.vllm_config.additional_config = {}
        self.vllm_config.additional_config["eplb_config"] = get_eplb_config_dict(self.od_config)

    def _maybe_prepare_eplb_runtime(self) -> None:
        if not self._is_npu_eplb_target():
            self.dynamic_eplb = False
            return

        self._ensure_vllm_additional_config()

        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(self.vllm_config)

    def _maybe_init_dynamic_eplb(self) -> None:
        self.dynamic_eplb = False
        if not self._is_npu_eplb_target():
            return

        from vllm_ascend.ascend_config import get_ascend_config

        eplb_config = get_ascend_config().eplb_config
        self.dynamic_eplb = bool(eplb_config.dynamic_eplb)
        if not self.dynamic_eplb or getattr(self, "_eplb_initialized", False):
            return
        if self.model_runner.pipeline is None:
            return

        from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
        from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
        from vllm_ascend.eplb.core.eplb_worker import EplbProcess
        from vllm_ascend.eplb.eplb_updator import EplbUpdator
        from vllm_ascend.eplb.utils import model_register

        self.eplb_loader = D2DExpertWeightLoader()
        self.manager = Manager()
        self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
        self.eplb_process = EplbProcess(
            shared_dict=self.shared_dict,
            policy_type=eplb_config.eplb_policy_type,
            enable_d2d=True,
        )
        self.process = self.eplb_process._launch_process()
        self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)

        model_register(self.model_runner.pipeline)
        self.eplb_adaptor = VllmEplbAdaptor(model=self.model_runner.pipeline)
        self.eplb_loader.set_adator(self.eplb_adaptor)
        self.eplb_updator.set_adaptor(self.eplb_adaptor)
        self.eplb_updator.warm_up_eplb()
        self._eplb_initialized = True
        logger.info("Worker %s: Initialized dynamic EPLB for diffusion pipeline.", self.rank)

    def _shutdown_eplb_runtime(self) -> None:
        if getattr(self, "dynamic_eplb", False) and getattr(self, "eplb_updator", None) is not None:
            try:
                self.eplb_updator.shutdown()
            except Exception as exc:
                logger.warning("Worker %s: Failed to shutdown EPLB updater cleanly: %s", self.rank, exc)

        manager = getattr(self, "manager", None)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception as exc:
                logger.warning("Worker %s: Failed to shutdown EPLB manager cleanly: %s", self.rank, exc)

        try:
            from vllm_ascend.ascend_config import clear_ascend_config

            clear_ascend_config()
        except Exception:
            pass

        self._eplb_initialized = False
        self.dynamic_eplb = False

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        self._maybe_prepare_eplb_runtime()
        super().load_model(load_format=load_format, custom_pipeline_name=custom_pipeline_name)
        self._maybe_init_dynamic_eplb()

    def execute_model(self, req, od_config):
        if getattr(self, "dynamic_eplb", False):
            self.eplb_updator.forward_before()
            try:
                return super().execute_model(req, od_config)
            finally:
                self.eplb_updator.forward_end()
        return super().execute_model(req, od_config)

    def shutdown(self) -> None:
        self._shutdown_eplb_runtime()
        return super().shutdown()
