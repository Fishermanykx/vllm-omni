# Experimental native Wan VAE normalization

Use `model_config.lingbot_npu_vae_native_norm: true` with
`lingbot_vae_spatial_sharding: false`. Default is false. It is incompatible with
`lingbot_npu_vae_norm_silu` and other decoder fast paths. Spatial-padding can be
selected independently. Only original diffusers decoder `WanRMS_norm` modules
are patched; checkpoint names and SiLU remain unchanged.

The native `aclnnLinalgVectorNorm` reads BF16 input and accumulates/outputs FP32,
instead of reducing a previously materialized FP32 tensor. Division still casts
the input to FP32: this does **not** remove the full path's Cast kernel. The L2
clamp at `1e-12`, normalization-to-BF16 boundary, scalar multiplication, learned
gamma and zero bias order remain intact. Different reduction kernels can round
differently. This is opt-in and has not passed full-video quality acceptance.

## Ascend950PR observations, 2026-11-04

Installed torch 2.10 / torch_npu 2.10.0.post7 and CANN9.2 beta2 register RmsNorm,
AddRmsNorm, GroupNormSilu, and ordinary SiLU; no independent RMSNorm+SiLU schema
was found. The CANN API header inventory agrees with that limited finding.

- GroupNormSilu reduces spatial/group statistics and subtracts mean, unlike Wan.
- Native RmsNorm needs channel-last adaptation and does not reproduce the L2
  clamp: epsilon=0 yielded NaN for zero vectors. Correcting statistics separately
  retained finite outputs but was slower. Do not replace Wan with it directly.
- Explicit `torch_npu.npu_silu` is deprecated and fails on this A5 installation
  because that wrapper has no ACLNN implementation. `F.silu` already dispatches
  to `aclnnSilu`; keep it.
- Native BF16-input/FP32-output reduction gave approximately 8–10% improvement
  for isolated norm+SiLU samples. One sampled shape had a small rounding delta.
- Full checkpoint VAE decoding on one physical NPU, latent `[1,16,3,58,104]`,
  produced `[1,3,9,464,832]`. One warmup plus three measurements per configuration:
  reference 345.72ms vs candidate 332.91ms (about 3.7% lower). Thirty decoder norms
  were patched. Output max absolute difference 0.03515625, relative L2 0.00560.
  This is a synthetic latent and a VAE-only measurement, not video quality or
  end-to-end RTF acceptance. No four-card run was performed.

Raw data/scripts: `/mnt/share/y00958577/experiments/lingbot-a5-20261104/`
(`native-vae-results.json`, `native-vae-decode-results.json`,
`probe_vae_native.py`, `probe_vae_decode.py`). Profiling directories:
`profiles-vae-native/` and `profiles-vae-decode-native/`. Original norm+SiLU
Triton path remains experimental and disabled due to negative production-shape
performance; these measurements do not enable it.

## Next fusion boundary

Prefer an FP32 L2 denominator compatible with the original path, followed by a
small channel-first division/cast/scale/gamma/SiLU epilogue. Preserve intermediate
BF16 rounding. Profile and tune tile sizes before integrating the reduction.
Only expand to a complete norm+SiLU Triton kernel after local speed and decoder
accuracy have been checked. Neither AddRmsNorm nor GroupNormSilu can bypass these
semantic checks, and no distributed statistics are needed for spatial shards
that retain all channels.
