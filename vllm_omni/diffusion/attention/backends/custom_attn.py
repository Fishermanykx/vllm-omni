from collections.abc import Callable
from typing import Any

import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.utils.platform_utils import detect_device_type, is_rocm

logger = init_logger(__name__)

class CustomAttn(AttentionImpl):
    """
    Base class for custom attention forward.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.is_cuda = detect_device_type() == "cuda"
        self.is_npu = detect_device_type() == "npu"
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self) -> Callable:
        if is_rocm():
            return self.forward_hip
        elif self.is_cuda:
            return self.forward_cuda
        elif self.is_npu:
            return self.forward_npu
        else:
            return self.forward_native

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_npu(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)
