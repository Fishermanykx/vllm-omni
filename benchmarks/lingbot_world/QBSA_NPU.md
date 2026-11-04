# LingBot Eagle QBSA preparation (Ascend 950)

This experimental path reuses MindIE-SD's native **Ada-BSA KV-block estimator**
with Eagle Quant Block Sparse Attention. It is opt-in and currently eager-only.
CPU contract tests do not establish device availability, accuracy or speedup.

Add this to the stage's `model_config` in `configs/npu_usp4_eager.yaml`:

```yaml
lingbot_npu_qbsa_cdf: 0.85
```

Use `enforce_eager: true`, BF16 and the engine-owned paged cache. Host-owned
singleton lengths are enabled automatically. Omit the key or use `null` to retain
dense attention. A requested provider failure is reported rather than silently
running dense attention. This path changes video self attention; text cross
attention and KV storage remain unchanged. Dense KV is still retained.

## Provider contract

The integration targets the MindIE-SD checkout containing
`SparseBlockEstimate`'s `ascend950` implementation and the Eagle QBSA native
operator, plus the matching Python quantization helper. The older public
`get_estimate_mask` wrapper rejects A5; this adapter calls
`mindiesd.layers._custom_ops.sparse_block_estimate` directly. Check both installed
native kernels, not just the existence of Python symbols.

* Selection: BNSD BF16 Q/K, block size 128, stride 8, scale `head_dim**-0.5 / 8`,
  threshold 0.85, `row_sparse=1.0`, `causal=False`, sink/recent protection on.
  CDF is estimated probability mass, **not a fixed 85% block keep ratio**.
  Sink/recent protect boundary blocks, not the entire multi-frame sink window.
* Recompute selection for each attention call using its current Q/K. No mask
  caching across diffusion steps or newly appended chunks.
* The estimator returns an int8 keep mask with KV columns aligned to 32 blocks.
  Crop that padding to `ceil(K/128)` and make it contiguous for EQBSA.
* Quantization: reuse `_eagle_qbsa_quant_qkv`, Q/K INT8 in 64-token blocks,
  V per-channel FP8 E4M3. Pass V storage as int8 with its FP8 dtype descriptor.
* EQBSA: BNSD, `[128,128]` sparse blocks, `inner_precise=4`, BF16 output,
  actual Q and KV lengths passed separately. Current adapter supports batch 1,
  equal Q/K head counts and head dimension 128, including non-square tails.

Source contracts to check in the matching MindIE-SD checkout:
`mindiesd/layers/flash_attn/sparse_flash_attn_ada_bsa.py`,
`mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py`,
`mindiesd/layers/_custom_ops.py`,
`csrc/ops/sparse_block_estimate/op_kernel/arch35/`, and
`tests/ops/eagle_quant_block_sparse_attention/`.

## Pending NPU validation

Reserve four cards before running. Keep the same devices, checkpoint, official
case 03 input, seed 42, 464x832, 16 fps, SP4/TP1, four DMD steps, 16 chunks and
streaming VAE configuration throughout each comparison.

1. Validate native estimator and EQBSA at Q=4524, K=27144, H=10, D=128,
   plus short/tail cases. Compare selected masks with the native Ada estimator;
   check mask density and output finiteness. Provider installation may need
   rebuilding the isolated native operators, not merely the Python plugin.
2. First compare the original numeric dense path against QBSA CDF=0.85. Keep
   fused RMSNorm/AdaLN off to isolate sparse/quantized attention differences.
   CDF=1.0 is an optional diagnostic, not proof of dense numerical equivalence.
3. Separately compare the previous full optimization stack against that same
   stack plus QBSA. Prior RMSNorm/AdaLN changes already caused video drift;
   do not attribute it to QBSA or treat that output as accepted ground truth.
4. For each variant run one warmup, at least three unprofiled measurements and
   a separate steady-state profile. Include estimation, quantization and layout
   conversion overhead; verify native kernels on every rank. Collect actual
   mask density outside timed regions, without host scalar reads in the hot path.
5. Compare matching seeds' latents and decoded video across the whole rollout.
   Sparse/quantized attention is lossy; no quality threshold is assumed.
   Report end-to-end and steady-state RTF at 16 fps, together with per-session
   results and provider/code revisions. No NPU performance claim is made yet.
