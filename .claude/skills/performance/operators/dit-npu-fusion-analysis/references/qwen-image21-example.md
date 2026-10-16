# Qwen-Image 2.1 源码预审示例

依据本地 vLLM-Omni `5b14308ec8ca873179711f3600df9e8b6a88d828` 的模型源码，2026-10-16 检查。本文只演示 `source-only` 预审，不引用历史 profile 的热点时间或收益，也不将远端实验补丁当作此 checkout 已有能力。目标文件为 `vllm_omni/diffusion/models/qwen_image_21/qwen_image_21_transformer.py`；路径相对目标仓，使用时重新定位符号和平台 dispatch。

Qwen-Image 2.1 是本例的目标，不默认等同于 Turbo checkpoint。hidden size、heads、head_dim、FFN width、输入分辨率及 prefix 长度从实际 checkpoint/config 和请求获取，不能从模型名称推断。

## 先查已有工作，再划分本次增量

以 QwenImage21/Qwen-Image-2.1、shared modulation、NormRoPE、SwiGLU 和目标文件搜索 open/merged/closed PR。[Omni #8729](https://github.com/vllm-project/vllm-omni/pull/8729) 和 [#8017](https://github.com/vllm-project/vllm-omni/pull/8017) 是检索种子，不是当前状态、平台支持或验收保证。重新读取其 head diff、开关和测试，对照目标 checkout 及已安装 provider。

若已有提案覆盖算法，将本次工作标为复用、平台适配或契约修正，不重新包装为新融合。远端部署中已经启用的共享 modulation/RoPE 也应记入实际基线；不能再次把未优化源码中的计算计作新增收益。

## QKV → Q/K RMSNorm → RoPE：继续扩大融合边界

`QwenImage21Attention.forward` 的路径为：

```text
to_qkv → split(Q, K, V) → unflatten(B,S,H,D)
       → norm_q/norm_k → cast 到 value.dtype → Q/K RoPE
       → prefix cache / SP 分支 → Attention → to_out
```

- `to_qkv` 已是 `QKVParallelLinear`。不能建议“首次合并三个投影”；可继续检查 packed QKV 的 split view 是否被后端连续化，以及是否存在多余布局拷贝。
- Q/K 使用 `diffusion/layers/norm.py` 中的 `RMSNorm`，不是此文件的 `QwenImage21ZeroCenterRMSNorm`；后者有 `weight + 1` 语义，不能混用。沿 `forward_npu` 核对 `npu_rms_norm` 的实际版本、权重 dtype 和舍入，再确定复合实现是否等价。
- 此 checkout 的 NPU 路径使用 `_apply_qwen_image21_rotary_emb_native`：先转 FP32，按相邻通道对计算 `(x1*cos-x2*sin, x1*sin+x2*cos)`，最后转回输入 dtype；cos/sin 来自复数频率的实部/虚部。不能直接套用半通道配对的 NeoX RoPE，也不能先把频率降为 BF16。
- Norm 输出到 RoPE 输入的 dtype 边界是契约。把两个阶段合成一个 kernel 时，不得无意删除原 BF16 舍入；数学表达式相近不证明逐位一致。

比较“单独 RoPE → QK Norm+RoPE → packed QKV split+Norm+RoPE → 加 attention 输出布局”的边界。检查 MindIE-SD 的 wrapper/schema/plugin/kernel 和 vLLM-Ascend 的普通、SIMT、MRoPE 等 Triton 变体；逐项核对频率精度、通道配对、norm 归约/舍入、UB 与输出 stride。MRoPE 的 interleaved 标志不自动代表通道按相邻对旋转；带 V norm 的实现会增加模型语义，不能直接替换。

最小验证包含独立 Q/K 输出、非连续 packed 输入、prefill/decode、尾块及实际 dtype/shape。没有 profile 前，连续化、Cast 或 transpose 只能列为待查项，不能声称已发生或是热点。

## 共享 modulation：区分重复计算消除与融合

`QwenImage21Transformer2DModel.forward` 已在 block 循环外计算 `modulation = self.modulation(temb)`。`QwenImage21TransformerBlock.forward` 内仍拆分两组参数，`_modulate` 执行 token 行选择与 `1 + scale`，残差路径执行 `gate.tanh()`。

可检查是否将相同的行选择、scale 准备和 tanh 移到每步一次，再向各 block 传入结果。这是共享准备/重复计算消除，不是新增 vendor 融合算子。先与已有 PR、部署补丁查重；不要再把 modulation 的线性投影当作逐层重复计算。

`causal_condition` 下有真实 timestep 行和额外 t=0 行，`_select_modulation_rows` 根据 token mask 选择。prefill、decode、batch>1 和 SP 的 mask/广播范围分别核对；不能将 decode 的单行广播外推到全序列。

## LayerNorm、调制和 gated residual：不能照搬 add-RMSNorm

block 的 `img_norm1/img_norm2` 是无 affine 的 `nn.LayerNorm`，与 Q/K RMSNorm 不同。两条残差路径分别为：

```text
h1 = h0 + tanh(gate1) * Attention(LayerNorm(h0) * (1 + scale1))
h2 = h1 + tanh(gate2) * MLP(LayerNorm(h1) * (1 + scale2))
```

本例没有 shift。AdaLN/残差复合候选必须支持 LayerNorm 的均值/方差、无 affine、scale-only、正确 token 广播和 BF16 乘法后加法舍入；不能用 add-RMSNorm 替换。若融合跨过 h1，需确认所有消费者及返回的 residual。

原地策略应核对张量所有权。在新生成分支输出上做乘加和覆盖输入 hidden states 是不同方案；前者也可能仍为两个 kernel，只能先说明分配变化。FP16 分支的 clip 语义需保留。

## SwiGLU：双输入激活与更大 FFN 都要比较

`QwenImage21SwiGLUFeedForward.forward` 为：

```text
out(SiLU(gate_layer(h)) * proj(h))
```

gate/up 为独立 `ColumnParallelLinear`，输出投影为 `RowParallelLinear`；此 checkout 使用 `nn.SiLU` 和乘法，不能套用 HunyuanImage 已有 `SiluAndMul` 的结论。

比较双输入 SiLU×up、先拼接再调用融合激活、合并 gate/up 投影加 epilogue、完整 FFN。普通 `npu_swiglu` 的输入打包轴、通道顺序和 SiLU 后的 BF16 舍入需核对；额外 cat/copy 可能抵消收益。若已有精确双输入 Triton PR，优先适配其设备/格式 gate 和测试，不自行移除硬件限制。

更大 FFN 要检查权重重排/常驻内存、TP 分片、GEMM 累加和激活舍入、临时输出生命周期。算子微基准通过不等于整层或端到端通过。

## Attention、prefix cache 与布局：按分支证明

prefill 在 Q/K norm 与 RoPE 后保存 prefix K/V；原生 dtype 路径 clone prefix，避免保留完整 prefill storage。decode 仅计算 target，再读缓存；缓存 K 已经旋转，不能再次 norm/RoPE。FP8 K/V、V-only FP8 与原生 dtype 的存储/scale 契约分别处理。

`sp_prefix_len` 和 `sp_decode` 分支通过 joint tensors 交给 Attention；非 SP decode 才走相应 prefix 拼接。不要为方便大融合强行换成 AR paged-cache/slot-mapping ABI。

`_resolve_attn_path` 与模型 forward 根据 block-causal、padding 和 decode 选择 piecewise span、dense mask 或 key-padding 路径。`Attention(causal=False)` 不代表全程没有结构 mask。FA/FIA 或 head-major concat 候选要检查真实 mask、布局、stride、cache 消费者及后续连续化；布局变化必须消除实际拷贝，而不是将其转移到下游。

## 此示例应交付的结论

源码预审可交付已有工作关系、活跃分支、候选边界、契约冲突和最小验证计划；候选均为 `unmeasured`，优化验收为 `unvalidated`。有 profile 后再补热点归因，有设备实测后再记录生效、精度和 E2E 结果。按 ExecutionEvidence 契约保存本次版本与范围，不把历史 Qwen 用例的图片 hash 或性能继承为当前结果。
