import math
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from transformers.configuration_utils import PretrainedConfig
from vllm.config import VllmConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

from vllm_omni.utils.platform_utils import detect_device_type
if detect_device_type() == 'npu':
    from vllm_ascend.ascend_forward_context import set_forward_context as set_forward_context
else:
    from vllm.v1.worker.gpu_model_runner import set_forward_context

MODEL = "Qwen/Qwen-Image-Edit-2509"
MODEL_VL = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NUM_BATCHED_TOKEND = 10000

class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.device = detect_device_type()
        self.x = torch.randn([1, 64, 24, 64], dtype=torch.float16).to(self.device)
        self.cos = torch.randn([64, 32], dtype=torch.float32).to(self.device)
        self.sin = torch.randn([64, 32], dtype=torch.float32).to(self.device)
        self.is_neox_style = False
        self.layer = RotaryEmbedding(self.is_neox_style)

    def test_rope_forward(self):
        vllm_config = VllmConfig()
        model_config = OmniDiffusionConfig.from_kwargs(**{"model": MODEL})
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config

        with set_forward_context(None, vllm_config):
            result = self.layer.forward(self.x, self.cos, self.sin)

        self.assertEqual(result.shape, self.x.shape)