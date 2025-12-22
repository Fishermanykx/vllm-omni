from importlib.util import find_spec

import torch
import torch.nn as nn
from einops import rearrange, repeat
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)


class AdaLayerNorm(CustomOp):
    """
    AdaLayerNorm:
        out = layernorm(x) * (1 + scale) + shift
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool=False,
        eps: float=1e-6
    ) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.hidden_size = hidden_size

    def forward_cuda(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        scale = (1 + scale.unsqueeze(1))
        shift = shift.unsqueeze(1)
        return torch.nn.functional.layer_norm(
                x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps), gate.unsqueeze(1)

    def forward_hip(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_native(x, mod_params)
    
    def forward_npu(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor
    ) -> torch.Tensor:
        import torch_npu
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        scale = (1 + scale.unsqueeze(1))
        shift = shift.unsqueeze(1)
        return torch_npu.npu_layer_norm_eval(
                x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps), gate.unsqueeze(1)

    def forward_native(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor
    ) -> torch.Tensor:
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        shift_result = shift.unsqueeze(1)
        scale_result = scale.unsqueeze(1)
        gate_result = gate.unsqueeze(1)

        return nn.LayerNorm(self.hidden_size, elementwise_affine=self.elementwise_affine, eps=self.eps)(x) * (1 + scale_result) + shift_result, gate_result
