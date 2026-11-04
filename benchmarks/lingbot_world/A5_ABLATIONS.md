# LingBot Ascend 950 online MXFP8 and fusion experiments

Use eager SP4/TP1, BF16 inputs/outputs and an explicitly reserved group of four
devices. The switches below are independent and disabled by default. They are
experimental paths, not accepted full-video accuracy or performance guarantees.

```yaml
model_config:
  lingbot_npu_host_lengths: true
  lingbot_npu_online_quant: mxfp8
  lingbot_vae_spatial_sharding: false
```

Online quantization uses the existing `DiffusionMXFP8Config` NPU implementation:
BF16 checkpoint weights become MXFP8 during loading; activations use dynamic
microscaling at each Linear. All 320 physical attention/FFN linears in the
40-layer model participate. Conditioning/camera linears and time/gate
modulation retain their existing precision. This is not INT8 Linear quantization
and does not quantize paged KV storage. Do not combine this convenience switch
with an explicit `quantization_config`.

Additional `model_config` switches:

| Switch | Contract |
|---|---|
| `lingbot_npu_qbsa_cdf: 0.85` | Native Ada-BSA selection plus EQBSA; see `QBSA_NPU.md`. Its native INT8 Q/K and FP8 V contract is independent of Linear MXFP8. Requires both native custom-op libraries. |
| `lingbot_npu_fused_qkv: true` | TP1 eager BF16 packed QKV; across-5120 RMSNorm, explicit BF16 norm boundary, adjacent-pair Wan RoPE. V stays unchanged. Not Qwen per-head MRoPE. |
| `lingbot_npu_vae_spatial_padding: true` | Unsharded decoder only; moves symmetric spatial padding into Conv3d while preserving temporal/cache assembly. First encountered shape compares bitwise, caches fallback on mismatch/error. |
| `lingbot_npu_vae_norm_silu: true` | Unsharded decoder with original diffusers `WanRMS_norm`; ATen FP32 L2 reduction plus Triton scale/gamma/SiLU epilogue. Preserves parameter names. Rejects the semantically different patched `RMSNormVAE`. |

The two VAE switches reject simultaneous use of another decoder fastpath.
Shape-specific padding probes are screening checks; full-video validation is
still needed. Norm+SiLU and QKV fusion change floating-point evaluation and need
their own numerical checks. A successful kernel launch does not imply a speedup.
Keep any candidate that regresses at production shapes disabled in the final
stack. Gate fusion and gate quantization are separate decisions; these switches
do not quantize gate values.

For each candidate, first check production shapes and finite outputs against
the current numerical reference. Then compare complete fixed-seed videos,
including history-window sliding. Run one warmup and at least three unprofiled
sessions, with a separate steady-state profile. Record actual quantized weight
materialization, every rank's kernels, source/provider revisions, device IDs,
layout/conversion overhead and all failed attempts. No video-quality threshold
or RTF improvement is assumed in advance.
