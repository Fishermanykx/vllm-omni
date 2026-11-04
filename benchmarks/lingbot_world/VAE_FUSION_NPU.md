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
