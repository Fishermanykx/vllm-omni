import torch

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.cache.taylor_cache_backend import (
    HunyuanTaylorCacheManager,
    TaylorCacheBackend,
    TaylorCacheRuntimeConfig,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig


def test_selector_returns_taylor_cache_backend():
    backend = get_cache_backend("taylor_cache", DiffusionCacheConfig())
    assert isinstance(backend, TaylorCacheBackend)


def test_taylor_cache_backend_enable_refresh():
    class _Model:
        pass

    # Use the exact class name expected by backend enable() check.
    PipelineCls = type("HunyuanImage3Text2ImagePipeline", (), {})
    pipeline = PipelineCls()
    pipeline.model = _Model()

    backend = TaylorCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    assert backend.is_enabled()
    assert hasattr(pipeline, "_taylor_cache_manager")
    assert hasattr(pipeline.model, "_taylor_cache_manager")
    assert pipeline._taylor_cache_manager is pipeline.model._taylor_cache_manager

    backend.refresh(pipeline, num_inference_steps=20, verbose=False)
    manager = pipeline._taylor_cache_manager
    assert isinstance(manager, HunyuanTaylorCacheManager)
    assert manager.num_steps == 20


def test_taylor_cache_manager_forecast_path():
    runtime_config = TaylorCacheRuntimeConfig(
        interval=2,
        order=1,
        low_freqs_order=0,
        high_freqs_order=1,
    )
    manager = HunyuanTaylorCacheManager(runtime_config)
    manager.reset(num_steps=4)

    manager.set_step(0)
    assert manager.should_full_compute()
    hidden = torch.randn(2, 8, 16)
    manager.update_from_full(hidden, past_key_values="pkv0")

    manager.set_step(1)
    assert not manager.should_full_compute()
    pred_hidden, pred_pkv = manager.forecast()
    assert pred_hidden.shape == hidden.shape
    assert pred_pkv == "pkv0"
