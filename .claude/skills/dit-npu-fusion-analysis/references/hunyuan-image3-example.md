# HunyuanImage 3.0 源码预审示例

此示例以本地 vLLM Omni `ed102c65aa46407c999a8d24e383ba98383669bb` 为依据，2026-10-14 检查。未提供真实 profiling、目标部署版本或 NPU 实测，因此全部为 `source-only` / `unmeasured`。源码路径相对目标仓；迁移时重新定位符号及行号。

## 残差 + RMSNorm：复用已有融合入口

- 模型：`vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`，`HunyuanImage3DecoderLayer.forward`。
- 当前 attention 后：`h = residual + attn_output; residual = h; normalized = post_attention_layernorm(h); mlp(normalized); output = residual + mlp_output`。
- 现有层：`vllm_omni/diffusion/layers/norm.py`，`RMSNorm.forward_npu(x, residual)` 已调用 `torch_npu.npu_add_rms_norm` 并返回 normalized 与 updated residual；无 residual 参数时调用 `npu_rms_norm`。
- 候选设计：`normalized, residual = post_attention_layernorm(attn_output, residual)`，后续 MLP 与最终 residual add 保留。局部前后表达式等价的可能性明确，需核对输入 dtype、残差加法舍入、所有消费者及 NPU 精度。
- 分类：模型接线为 `adapter`；底层能力为源码 `already-fused`。不能写“需要新增 add_rms_norm 算子”，也不能断言当前运行已消费融合。
- 最小验证：目标形状、非连续输入、BF16/FP16 允许范围、returned residual 与最终 layer output 对照；profile 检查对应 Add 与独立 Norm 是否减少。没有 profile 前不设为性能 P0。

## QK Norm 与 RoPE：运算次序是契约

`HunYuanAttention.forward` 先做 QKV projection/split，随后按 `gen_image` 分支调用 `image_rope2d_emb` 或普通 `rotary_emb`，之后才在 `use_qk_norm` 条件下做 Q/K RMSNorm。

任何“QKV → Norm → RoPE”复合 kernel 都必须与这条“QKV → RoPE → Norm”路径逐项对照。带可学习 gamma 的 RMSNorm 与旋转通常不可交换，不能为使用现成复合算子改变次序。相同模型名、同为 RMSNorm 或同为 RoPE 都不是等价证明。候选若只支持相反次序，应 `blocked`；可分别保留 RoPE 与现有 Norm 较小边界。

核对 `HunYuanRotary2DEmbedder`、`apply_rotary_pos_emb` 的真实 cos/sin 生成与 half/interleaved 约定；不能把 MRoPE/token-aligned table 直接当作普通 positions。再沿平台 dispatch 确认当前 RoPE 是否已融合。

## MLP：避免重复建议已有 SiluAndMul

同文件 `HunYuanMLP` 使用 merged gate/up projection 与 `SiluAndMul`。这说明源码已有融合抽象，是否在当前 NPU 后端映射为 fused kernel，需要继续追踪 vLLM/vLLM-Ascend 的实际 dispatch。不能仅凭 `silu` 热点名字就新增 `npu_swiglu`。MoE 分支走 `HunYuanSparseMoeBlock`，不能拿 dense MLP 结论覆盖 routed experts。

## Attention：按 mask 和 cache 分支核对

`ImageKVCacheManager.forward` 先检查 scheduler-paged KV 条件，满足时走 `_forward_paged`；dense 路径要求 attention mask，缺失时直接报错，再进入 `_forward_dense_legacy`。后者含 K/V repeat、layout 和 SDPA 相关处理。先沿 backend/capability 路径检查有效 GQA 支持，再考虑现成 FA/FIA。

需要生产参数证明 `Nq/Nkv`、prefix/gen 长度、mask/causal、cache 生命周期、seq layout 及转换成本。paged 与 dense 路径不能共用未经核对的“直接替换”结论；不能假设 mask 为 None 就存在普通无 mask dense 路径。避免因 FIA 名称相似而套用 HunyuanImage 3.5 的量化 cache ABI。

## 此示例应产出的结论

存在可复用的 residual+RMSNorm 接线机会；QK 复合融合需严格核对次序；MLP/Norm 已有融合抽象；Attention 需按分支和后端继续审计。当前只能列候选与验证方法，不能声称这些是瓶颈、部署支持或已有收益。
