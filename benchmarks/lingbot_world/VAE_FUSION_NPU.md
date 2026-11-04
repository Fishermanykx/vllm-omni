# LingBot Wan VAE fusion on Ascend950PR

## Epilogue adaptation

Enable `lingbot_npu_vae_norm_silu: true` in model_config, with
`lingbot_vae_spatial_sharding: false`. It stays default-off and is mutually
exclusive with native-norm/other decoder fast paths.

This adapts the rounding contract of the existing shared CUDA/XPU VAE epilogue
(`diffusion/layers/vae_norm_epilogue.py`) to NPU persistent channel-first work.
The original FP32 ATen reduction and native SiLU remain. One persistent Triton
kernel folds divide, BF16 cast, scale, gamma and zero bias. It uses blocks of
4096 spatial elements instead of the earlier 512-element global launch grid.
Noncontiguous/unsupported inputs, autograd and compilation use the original
norm+SiLU path. Decoder activation and norm hooks are not bypassed.

`tl.div_rn` alone differed at BF16 rounding boundaries on the tested Ascend
backend. An explicit FMA residual corrects the quotient before BF16 conversion.
This is not a formal proof across all FP32 values/toolkit versions. Preserve
the explicit BF16 boundaries and `enable_fp_fusion=False`.

## Evidence, 2026-11-04

Single physical NPU6, checkpoint VAE weights, seeded latent `[1,16,3,58,104]`,
output `[1,3,9,464,832]`, BF16, one warmup plus three unprofiled repetitions.
Integrated epilogue: baseline 346.05ms vs candidate 291.52ms (~15.8% lower).
Output was bit-identical in this sample. Exploratory epilogue without the final
layout guard measured ~20% lower, with three seeds producing identical outputs;
do not substitute that less restrictive result for the integrated measurement.
Twenty-nine decoder norm+activation pairs were installed; individual unsupported
calls can still fall back. A separate profile was collected after timing.

Artifacts under `/mnt/share/y00958577/experiments/lingbot-a5-20261104/`:

- `vae-epilogue-integrated-results.json`, `probe_vae_epilogue_integrated.py`.
- `profiles-vae-epilogue-integrated/` (CANN kernel details and Chrome trace).
- `vae-fusion-results.json`, `vae-exact-epi-results.json`,
  `vae-exact-epi-uncorrected-results.json`, `vae-rounding-results.json` preserve
  initial tile/backend screening, failed compiler configurations and rounding
  investigation. Screening timings are not a full-model performance claim.

No four-card/full-DiT inference or video-quality acceptance is claimed. The
adapter remains opt-in; the previous native reduction variant and older VAE
reports are separate configurations and must not be combined as additive gains.

## Experimental full boundary

With `lingbot_npu_vae_norm_silu: true`, explicitly set
`lingbot_npu_vae_norm_backend: full` to also fuse the channel reduction and SiLU.
Omitting the backend selects `epilogue`. Supplying a backend without enabling
the norm path is an error. Both modes remain opt-in; neither changes defaults.

The full kernel uses FP32 square/sum/sqrt and the original L2 clamp, then
BF16-rounded division, scale, gamma and SiLU. It supports tested channels
96/192/384; other channel counts use epilogue. Spatial blocks are 64 positions,
with SIMD for C96/C192 and SIMT for C384. C384 SIMD exceeded register capacity
in screening. Tensor layout/autograd guards remain shared with epilogue.

Changing reduction order produced approximately 0.56% relative L2 difference in
checkpoint VAE outputs for three seeded synthetic inputs. Exploratory full
timings were ~254ms vs ~346ms baseline, before the final layout guard. Treat
the integrated JSON as authoritative for the committed implementation's timing.
This backend is not recommended as a numerically exact replacement and has
not passed video-quality acceptance. The original full kernel's compilation
failures and all exploratory timings remain in the artifacts.

Final integrated verification artifacts: `vae-fusion-integrated-results.json`,
`probe_vae_fusion_integrated.py`, `profiles-vae-fusion-integrated-epilogue/`,
`profiles-vae-fusion-integrated-full/`, and `vae-fusion-npu-tests.log` in the same
server experiment directory. These are VAE-only, single-card experiments;
the full serving pipeline/backend configuration was not exercised on four cards.

Final guarded implementation, same single-card VAE case: reference 345.88ms,
epilogue 292.15ms (15.5% lower), full 271.94ms (21.4% lower). Each has one warmup
and three measurements. Epilogue decoder outputs were bit-identical for seeds
42/43/44; full relative L2 was 0.00561–0.00567. Baseline repeated against itself
was bit-identical. Full is only ~6.9% faster than epilogue, with an additional
numerical change; prefer epilogue until video-level quality is accepted.
