import math
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from transformers.configuration_utils import PretrainedConfig
from vllm.config import VllmConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

from vllm_omni.utils.platform_utils import detect_device_type
if detect_device_type() == 'npu':
    from vllm_ascend.ascend_forward_context import set_forward_context as set_forward_context
else:
    from vllm.v1.worker.gpu_model_runner import set_forward_context

MODEL = "Qwen/Qwen-Image-Edit-2509"
MODEL_VL = "Qwen/Qwen2.5-VL-3B-Instruct"

class TestAdaLayerNorm(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.device = detect_device_type()
        self.hidden_size = 32
        self.elementwise_affine = False
        self.eps = 1e-6

        self.x = torch.randn([1, 128, 32], dtype=torch.float32).to(self.device)
        self.mod_params = torch.randn([1, 96], dtype=torch.float32).to(self.device)
        _, _, self.gate = self.mod_params.chunk(3, dim=-1)
        self.gate = self.gate.unsqueeze(1)

        self.modulated = torch.randn([1, 128, 32], dtype=torch.float32).to(self.device)

        self.layer = AdaLayerNorm(self.hidden_size, self.elementwise_affine, self.eps)

    def test_adalayernorm_forward(self):
        vllm_config = VllmConfig()
        model_config = OmniDiffusionConfig.from_kwargs(**{"model": MODEL})
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config

        with set_forward_context(None, vllm_config):
            result_modulated, result_gate = self.layer.forward(self.x, self.mod_params)

        self.assertEqual(result_modulated.shape, self.modulated.shape)
        self.assertEqual(result_gate.shape, self.gate.shape)