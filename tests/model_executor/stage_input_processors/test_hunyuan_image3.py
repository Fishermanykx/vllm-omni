# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanImage3 stage input processor."""

from types import SimpleNamespace

from PIL import Image

from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
    _first_source_image,
    ar2diffusion,
)

pytestmark = []


def _source_output(token_ids: list[int]):
    return SimpleNamespace(
        outputs=[SimpleNamespace(cumulative_token_ids=token_ids, text="")],
        multimodal_output=None,
    )


def test_first_source_image_accepts_img2img_key():
    image = Image.new("RGB", (16, 16))

    assert _first_source_image({"img2img": image}) is image


def test_ar2diffusion_forwards_img2img_when_multimodal_required():
    image = Image.new("RGB", (16, 16))
    stages = [SimpleNamespace(engine_outputs=[_source_output([1, 2, 3])])]

    result = ar2diffusion(
        stages,
        [0],
        [{"prompt": "keep the same person", "multi_modal_data": {"img2img": image}}],
        requires_multimodal_data=True,
    )

    assert result[0]["pil_image"] is image
    assert result[0]["multi_modal_data"]["image"] is image


def test_ar2diffusion_drops_image_when_multimodal_not_required():
    image = Image.new("RGB", (16, 16))
    stages = [SimpleNamespace(engine_outputs=[_source_output([1, 2, 3])])]

    result = ar2diffusion(
        stages,
        [0],
        [{"prompt": "keep the same person", "multi_modal_data": {"img2img": image}}],
        requires_multimodal_data=False,
    )

    assert "pil_image" not in result[0]
    assert "multi_modal_data" not in result[0]
