# HunyuanImage-3.0-Instruct

This example runs HunyuanImage-3.0-Instruct offline through the unified deploy
YAMLs under `vllm_omni/deploy/`.

## Default Configs

| File | Use case | Notes |
| :--- | :--- | :--- |
| `vllm_omni/deploy/hunyuan_image3.yaml` | AR + DiT | Default for `text2img` and `img2img`. Uses `mode` to select text-to-image or image-editing. |
| `vllm_omni/deploy/hunyuan_image3_ar.yaml` | AR only | Default for `img2text` and `text2text`. |
| `vllm_omni/deploy/hunyuan_image3_dit.yaml` | DiT only | Standalone DiT deployment/inference. |

## Run Examples

Text to image:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --prompts "A cute cat sitting on a windowsill watching the sunset"
```

Image editing:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality img2img \
  --image-path /path/to/image.png \
  --prompts "Make the petals neon pink"
```

Image to text:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality img2text \
  --image-path /path/to/image.jpg \
  --prompts "Describe the content of the picture."
```

Text to text:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2text \
  --prompts "What is the capital of France?"
```

Override the deploy YAML explicitly:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config vllm_omni/deploy/hunyuan_image3.yaml \
  --prompts "A cute cat"
```

## Key Arguments

| Argument | Description |
| :--- | :--- |
| `--deploy-config` | Preferred config path for unified deploy YAMLs. |
| `--stage-configs-path` | Legacy stage config path, kept only for compatibility. |
| `--modality` | One of `text2img`, `img2img`, `img2text`, `text2text`. |
| `--steps` | Number of diffusion inference steps for image generation. |
| `--guidance-scale` | Classifier-free guidance scale for image generation. |
| `--height`, `--width` | Output image size for `text2img`. |

## Notes

The unified AR+DiT deploy config enables AR-to-DiT KV cache reuse. Platform
overrides for CUDA/NPU/XPU are folded into the deploy YAML, so the older
HunyuanImage3 files under `model_executor/stage_configs/` and
`platforms/*/stage_configs/` are no longer needed.
