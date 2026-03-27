# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization config for diffusion transformers on Ascend NPU.

Ascend NPU does not support the standard vLLM CUDA FP8 execution path. This
config provides an online MXFP8-style path for diffusion linear layers by
quantizing BF16/FP16 weights after loading and dynamically quantizing
activations at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform

if current_omni_platform.is_npu():
    import torch_npu
else:
    torch_npu = None

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["dynamic"]


def create_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader,
    params_dtype: torch.dtype,
) -> torch.nn.Parameter:
    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=params_dtype,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


def _get_mxfp8_scale_dtype():
    if torch_npu is None:
        raise RuntimeError("FP8 online quantization requires torch_npu.")
    scale_dtype = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
    if scale_dtype is None:
        raise RuntimeError("MXFP8 scale dtype is not available in the current torch_npu runtime.")
    return scale_dtype


class DiffusionNPUFP8Config(QuantizationConfig):
    """Online FP8 config for diffusion models on Ascend NPU.

    Internally this uses MXFP8-style quantization on NPU because the regular
    CUDA FP8 execution path is not available on Ascend.
    """

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        group_size: int = 32,
    ) -> None:
        super().__init__()

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionNPUFP8Config":
        activation_scheme = cls.get_from_keys_or(config, ["activation_scheme"], "dynamic")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        group_size = cls.get_from_keys_or(config, ["group_size"], 32)
        return cls(
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            group_size=group_size,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not current_omni_platform.is_npu():
                raise NotImplementedError("The current platform is not supported for NPU FP8 online quant.")
            return NPUFP8OnlineLinearMethod(self)
        return None


class NPUFP8OnlineLinearMethod(LinearMethodBase):
    """Online MXFP8 linear method for Ascend NPU."""

    def __init__(self, quant_config: DiffusionNPUFP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        weight = create_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            params_dtype=params_dtype,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if torch_npu is None:
            raise RuntimeError("FP8 online quantization requires torch_npu.")
        if not hasattr(torch_npu, "npu_dynamic_mx_quant") or not hasattr(torch_npu, "npu_quant_matmul"):
            raise RuntimeError("Current torch_npu runtime does not provide MXFP8 quantization APIs.")
        _get_mxfp8_scale_dtype()

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        qweight, weight_scale = torch_npu.npu_dynamic_mx_quant(layer.weight, dst_type=torch.float8_e4m3fn)
        qweight = qweight.t().contiguous()
        if weight_scale.dim() == 2:
            n_dim, k_dim = weight_scale.shape
            if k_dim % 2 == 0:
                weight_scale = weight_scale.reshape(n_dim, k_dim // 2, 2)
            weight_scale = weight_scale.transpose(0, 1).contiguous()

        replace_parameter(layer, "weight", qweight)
        weight_scale_param = torch.nn.Parameter(weight_scale, requires_grad=False)
        if "weight_scale" in layer._parameters:
            replace_parameter(layer, "weight_scale", weight_scale_param)
        else:
            layer.register_parameter("weight_scale", weight_scale_param)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch_npu is None:
            raise RuntimeError("FP8 online quantization requires torch_npu.")

        scale_dtype = _get_mxfp8_scale_dtype()
        orig_shape = x.shape
        orig_dtype = x.dtype
        x = x.reshape(-1, orig_shape[-1])
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=scale_dtype,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=scale_dtype,
            bias=bias,
            output_dtype=orig_dtype,
            group_sizes=[1, 1, self.quant_config.group_size],
        )
        return output.reshape(*orig_shape[:-1], -1)
