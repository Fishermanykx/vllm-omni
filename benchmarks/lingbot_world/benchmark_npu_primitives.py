# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free NPU screens; reports errors and timings, not model acceptance."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sp-degree", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--include-adaln", action="store_true", help="Requires MindIE-SD FP32 AdaLN support")
    parser.add_argument("--history-staging", action="store_true", help="Measure persistent history buffers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")

    if args.history_staging:
        os.environ["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] = "1"

    import torch
    import torch_npu

    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
        ARDiffusionPagedLayerInputs,
        paged_write_attn,
    )

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    device = torch.device("npu", args.device)
    dtype = torch.bfloat16

    def measure(operation):
        for _ in range(args.warmup):
            operation()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            operation()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"samples_ms": samples, "median_ms": statistics.median(samples)}

    def errors(reference, candidate):
        diff = (reference.float() - candidate.float()).abs()
        return {
            "bit_exact": bool(torch.equal(reference, candidate)),
            "max_abs": diff.max().item(),
            "max_relative": (diff / reference.float().abs().clamp_min(1e-8)).max().item(),
            "mean_abs": diff.mean().item(),
            "candidate_finite": bool(torch.isfinite(candidate).all()),
        }

    # Match the flat-projection arithmetic in _LingBotRMSNorm. No per-head
    # reshape or distributed reduction is used in this TP1 provider screen.
    x = torch.randn(1, 4680 // args.sp_degree, 5120, device=device, dtype=dtype)
    weight = torch.randn(5120, device=device, dtype=dtype)

    def native_norm():
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).sum(-1, keepdim=True) / x.shape[-1] + 1e-6)
        return (xf / rms * weight.float()).to(dtype)

    def fused_norm():
        return torch_npu.npu_rms_norm(x, weight, epsilon=1e-6)[0]

    q_len, kv_len, block_size = 4680, 18 * 1560, 16
    heads, dim = 40 // args.sp_degree, 128
    capacity = ((kv_len + block_size - 1) // block_size + 1) * block_size
    shape = (capacity, heads, dim)
    key_pool = torch.randn(shape, device=device, dtype=dtype)
    value_pool = torch.randn_like(key_pool)
    query = torch.randn(q_len, heads, dim, device=device, dtype=dtype)
    keys, values = torch.randn_like(query), torch.randn_like(query)
    inputs = ARDiffusionPagedLayerInputs(
        layer_idx=torch.tensor(0),
        key_pool=key_pool,
        value_pool=value_pool,
        block_size=block_size,
        seq_len=q_len,
        video_slots=torch.arange(kv_len - q_len, kv_len, device=device),
        action_slots=torch.empty(0, device=device, dtype=torch.long),
        block_table=torch.arange(capacity // block_size, device=device, dtype=torch.int32).unsqueeze(0),
        query_start_loc=torch.tensor([0, q_len], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([kv_len], device=device, dtype=torch.int32),
        max_query_len=q_len,
        max_seq_len=capacity,
        stage_key=torch.empty(0, device=device, dtype=dtype),
        stage_value=torch.empty(0, device=device, dtype=dtype),
    )
    host_inputs = inputs._replace(npu_lengths=(q_len, kv_len))

    def native_attention():
        return paged_write_attn(inputs, query, keys, values, None, None, dim**-0.5)

    def host_attention():
        return paged_write_attn(host_inputs, query, keys, values, None, None, dim**-0.5)

    staged_inputs = (
        host_inputs._replace(
            stage_key=torch.empty_like(key_pool),
            stage_value=torch.empty_like(value_pool),
            stage_first_block=(kv_len - q_len) // block_size,
        )
        if args.history_staging
        else None
    )

    def staged_attention(reuse):
        return paged_write_attn(staged_inputs._replace(reuse_history=reuse), query, keys, values, None, None, dim**-0.5)

    with torch.inference_mode():
        result = {
            "scope": "synthetic TP1 primitives; no model or end-to-end acceptance",
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
            "device": str(device),
            "sp_degree_shape": args.sp_degree,
            "attention_shape": {"query": list(query.shape), "kv_len": kv_len},
            "rmsnorm_errors": errors(native_norm(), fused_norm()),
            "attention_errors": errors(native_attention(), host_attention()),
            "native_rmsnorm": measure(native_norm),
            "fused_rmsnorm": measure(fused_norm),
            "native_attention": measure(native_attention),
            "host_lengths_attention": measure(host_attention),
        }
        if args.history_staging:
            staged_attention(False)  # initialize all history before retaining it
            keys.add_(0.125)  # exercise a changed current chunk on the retained window
            result["staged_attention_errors"] = errors(native_attention(), staged_attention(True))
            result["full_staging_attention"] = measure(lambda: staged_attention(False))
            result["reuse_staging_attention"] = measure(lambda: staged_attention(True))
            result["staging_bytes"] = 2 * key_pool.numel() * key_pool.element_size()
        if args.include_adaln:
            from mindiesd import layernorm_scale_shift

            # TP1/SP preparation makes timestep modulation tokenwise after
            # sharding: [B*S, 1, D], with FP32 normalization and scale/shift.
            norm = torch.nn.LayerNorm(5120, eps=1e-6, elementwise_affine=False).to(device)
            value = x.float().reshape(-1, 1, 5120)
            scale, shift = torch.randn_like(value[:, 0]), torch.randn_like(value[:, 0])

            def native_adaln():
                return (norm(value) * (1 + scale[:, None]) + shift[:, None]).to(dtype)

            def fused_adaln():
                return layernorm_scale_shift(norm, value, scale, shift, fused=True).to(dtype)

            result["adaln_errors"] = errors(native_adaln(), fused_adaln())
            result["native_adaln"] = measure(native_adaln)
            result["fused_adaln"] = measure(fused_adaln)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
