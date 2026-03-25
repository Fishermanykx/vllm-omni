# Image-To-Video

This example demonstrates how to generate videos from images using Wan2.2 Image-to-Video models with vLLM-Omni's offline inference API.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num-frames`: Number of frames (default 81).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--num-inference-steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

## Wan2.2 I2V + LightX2V (Offline Conversion)

If you want to run Wan2.2 I2V through a LightX2V-converted Diffusers directory in vLLM-Omni,
use the offline conversion route below.

### Required assets

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- LoRA weights: `lightx2v/Wan2.2-Distill-Loras`
- LightX2V converter: `tools/convert/converter.py`

### Step 1: Convert high/low noise DiT weights

```bash
python converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file

python converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

### Step 2: Assemble a final Diffusers-style directory

```bash
python tools/wan22/assemble_lightx2v_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --high-noise-weight /tmp/wan22_lightx2v/high_noise_out \
  --low-noise-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers \
  --asset-mode symlink \
  --overwrite
```

### Step 3: Run offline inference

```bash
python image_to_video.py \
  --model /path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers \
  --image /path/to/input.jpg \
  --prompt "A cat playing with yarn" \
  --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --height 480 \
  --width 832 \
  --flow-shift 12 \
  --sample-solver euler \
  --guidance-scale 1.0 \
  --guidance-scale-high 1.0 \
  --boundary-ratio 0.875
```

Notes:

- This route avoids runtime LoRA loading changes in vLLM-Omni.
- Output quality/speed depends on chosen LightX2V LoRA and sampling params.
- For native diffusion LoRA behavior in vLLM-Omni, see `docs/user_guide/diffusion/lora.md`.
