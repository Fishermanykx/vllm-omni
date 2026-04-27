# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for HunyuanImage3: AR -> Diffusion transition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)
CFG_TEXT_SUFFIX = "__cfg_text"
_AR_TOKENIZER_CACHE: dict[str, Any] = {}


def _first_source_image(mm_data: Any) -> Any:
    """Get the first source image from common multimodal keys."""
    if not isinstance(mm_data, dict):
        return None

    for key in ("image", "img2img", "images"):
        image = mm_data.get(key)
        if image is None:
            continue
        if isinstance(image, list):
            return image[0] if image else None
        return image
    return None


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.cumulative_token_ids
        generated_text = getattr(output, "text", "") or ""

        if not generated_text and generated_token_ids:
            tokenizer = _resolve_ar_tokenizer(stage_list[source_stage_id])
            if tokenizer is not None:
                try:
                    generated_text = tokenizer.decode(list(generated_token_ids), skip_special_tokens=False)
                except Exception as exc:
                    logger.warning(
                        "[ar2diffusion] Failed to decode AR tokens for request %d: %s",
                        i,
                        exc,
                    )

        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        text_prompt = original_prompt.get("user_prompt") or original_prompt.get("prompt", "")
        use_system_prompt = original_prompt.get("use_system_prompt")

        logger.info(
            "[ar2diffusion] Request %d: AR generated %d tokens, text length=%d, target size=%dx%d",
            i,
            len(generated_token_ids),
            len(generated_text),
            height,
            width,
        )

        trigger_tag = original_prompt.get("trigger_tag")
        if trigger_tag and generated_text and not generated_text.startswith(trigger_tag):
            generated_text = trigger_tag + generated_text

        token_tensor = torch.tensor(generated_token_ids, dtype=torch.long)

        diffusion_input: dict[str, Any] = {
            "prompt": text_prompt,
            "height": height,
            "width": width,
            "extra": {
                "ar_token_ids": token_tensor,
                "ar_generated_text": generated_text,
            },
        }

        if use_system_prompt is not None:
            diffusion_input["use_system_prompt"] = use_system_prompt

        if requires_multimodal_data:
            prompt_image = _first_source_image(original_prompt.get("multi_modal_data"))
            if prompt_image is not None:
                diffusion_input["pil_image"] = prompt_image
                diffusion_input["multi_modal_data"] = {"image": prompt_image}

        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                diffusion_input["extra"]["ar_multimodal_output"] = mm_output

        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs


def _resolve_ar_tokenizer(stage_client: Any) -> Any:
    """Best-effort resolution of the AR stage tokenizer."""
    model_path = None
    vllm_config = getattr(stage_client, "vllm_config", None)
    if vllm_config is not None:
        model_cfg = getattr(vllm_config, "model_config", None)
        if model_cfg is not None:
            model_path = getattr(model_cfg, "tokenizer", None) or getattr(model_cfg, "model", None)
    if model_path is None:
        model_path = getattr(stage_client, "model", None) or getattr(stage_client, "model_name", None)
    if not model_path:
        return None
    if model_path in _AR_TOKENIZER_CACHE:
        return _AR_TOKENIZER_CACHE[model_path]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("[ar2diffusion] Could not load tokenizer from %r: %s", model_path, exc)
        _AR_TOKENIZER_CACHE[model_path] = None
        return None
    _AR_TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list:
    """Expand user prompt into companion prompts for HunyuanImage3 CFG."""
    from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

    if not isinstance(prompt, dict):
        return []

    modalities = prompt.get("modalities", [])
    if "image" not in modalities:
        return []

    cfg_token = "<cfg>"
    pos_prompt_text = prompt.get("prompt")
    user_text = prompt.get("user_prompt")
    explicit_neg = _get_negative_prompt(prompt, sampling_params)

    if explicit_neg:
        neg_prompt = explicit_neg
    elif pos_prompt_text and user_text and user_text in pos_prompt_text:
        neg_prompt = pos_prompt_text.replace(user_text, cfg_token, 1)
    elif pos_prompt_text:
        neg_prompt = pos_prompt_text
    else:
        neg_prompt = "<|startoftext|>"

    neg_prompt_dict: dict[str, Any] = {
        "prompt": neg_prompt,
        "modalities": prompt.get("modalities", []),
    }
    if "multi_modal_data" in prompt:
        neg_prompt_dict["multi_modal_data"] = prompt["multi_modal_data"]
    if "height" in prompt:
        neg_prompt_dict["height"] = prompt["height"]
    if "width" in prompt:
        neg_prompt_dict["width"] = prompt["width"]
    if "use_system_prompt" in prompt:
        neg_prompt_dict["use_system_prompt"] = prompt["use_system_prompt"]

    return [
        ExpandedPrompt(
            prompt=neg_prompt_dict,
            role="cfg_text",
            request_id_suffix=CFG_TEXT_SUFFIX,
            sampling_params_override={"max_tokens": 1},
        ),
    ]


def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: Any,
    target_device: Any | None = None,
) -> dict[str, Any]:
    """Collect KV caches for CFG companion requests."""
    result: dict[str, Any] = {}

    for role, companion_rid in cfg_request_ids.items():
        try:
            data, size = kv_transfer_manager.receive_kv_cache_for_request(companion_rid, target_device)
            if data and "layer_blocks" in data:
                layer_blocks = data["layer_blocks"]
                kv_obj = SimpleNamespace(**layer_blocks)
                result[f"{role}_past_key_values"] = kv_obj
                if "metadata" in data:
                    result[f"{role}_kv_metadata"] = data["metadata"]
                logger.info(
                    "Collected CFG KV cache for role=%s, rid=%s, size=%d bytes",
                    role,
                    companion_rid,
                    size,
                )
            else:
                logger.warning(
                    "Failed to collect CFG KV cache for role=%s, rid=%s",
                    role,
                    companion_rid,
                )
        except Exception as e:
            logger.exception(
                "Error collecting CFG KV cache for role=%s, rid=%s: %s",
                role,
                companion_rid,
                e,
            )

    return result


def _get_negative_prompt(
    prompt: dict[str, Any],
    sampling_params: Any,
) -> str:
    """Resolve the negative prompt from prompt dict or sampling params."""
    neg = prompt.get("negative_prompt")
    if neg:
        return neg

    if hasattr(sampling_params, "extra_args") and sampling_params.extra_args:
        neg = sampling_params.extra_args.get("negative_prompt")
        if neg:
            return neg

    return ""
