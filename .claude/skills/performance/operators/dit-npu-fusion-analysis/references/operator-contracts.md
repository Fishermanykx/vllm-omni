# 昇腾算子检索与契约核对

## 来源与版本

以下是 2026-10-14 检索时的入口，不是当前部署的支持矩阵。每次分析仍需按用户安装版本定位文档/tag；保留版本、URL、访问日期以及未能获取的正文。

- [MindIE 3.0.0 Core Acceleration APIs](https://www.hiascend.com/document/detail/en/mindie/300/viewframe/sddev/docs/en/features/core_layers.md)：官方 API 文档，涵盖 attention、Norm、RoPE 和 activation。检索时可读正文。
- [MindIE-SD 官方源码](https://github.com/Ascend/MindIE-SD)：核对目标 release/tag 下 wrapper、底层实现和测试；分支名不等于发布包版本。
- [TorchNPU 26.1.X 使用导读](https://www.hiascend.com/document/detail/zh/Pytorch/2610/index/index.html)：从目标版本索引查支持产品、依赖和 API 变更。
- [torch_npu 自定义 API 索引](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu_list.md)：发现候选，再进入单算子页。
- [npu_add_rms_norm](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md)：检索摘要显示三 Tensor 输出；本次直接访问超时，具体约束须在使用时读取正文或对应包源码。
- [历史 CANN 8.0 RC1 alpha003 npu_rotary_mul](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC1alpha003/apiref/fmkadptapi/ptaoplist_000717.html)：只作历史契约示例，该页面的维度/对齐要求不能泛化到新版。
- [Ascend op-plugin](https://gitcode.com/Ascend/op-plugin)：按 tag 查 `docs/zh/custom_APIs/torch_npu/` 与算子实现。
- [vLLM-Ascend Triton 实现](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton)：检索部署版本中的 `linearnorm/`、相关模型子目录及 CustomOp 注册/调用处。main 只用于发现候选，不作为部署可用性证明。

版本不明时，API 名只能列作检索线索；不能生成声称可直接执行的调用示例。文档与实际安装包冲突时记录冲突，优先复现目标包实际行为，不能静默选较宽松的一方。

## 模式到候选的检索路线

| 真实子图 | 检索词/候选 | 关键排除条件 |
| --- | --- | --- |
| add → RMSNorm | `npu_add_rms_norm`、MindIE-SD `add_rms_norm` | 残差消费者、tuple 顺序、cast/累积精度、广播 |
| RMSNorm | `npu_rms_norm`、MindIE-SD `RMSNorm` | 已有平台融合、weight dtype/乘法精度、eps |
| LayerNorm → scale/shift | MindIE-SD `layernorm_scale_shift` | 是 LayerNorm 还是 RMSNorm、是否 `1+scale`、广播/affine |
| add → LayerNorm | MindIE-SD `add_layer_norm` | residual 输出、bias/affine、支持的维度 |
| split/chunk → SiLU → mul | `npu_swiglu`、现有 `SiluAndMul` dispatch | gate/up 次序、dim、切分布局、是否已融合 |
| RoPE | `npu_rotary_mul`、`npu_apply_rotary_pos_emb`、MindIE-SD `rotary_position_embedding` | half/interleaved、partial rotary、MRoPE 位置、cos/sin 布局、次序 |
| QKV + Norm + RoPE | MindIE-SD `norm_rope_concat`；vLLM-Ascend `linearnorm/` 中 split QKV/RMSNorm/RoPE、MRoPE、SIMT/TP 变体 | registry/schema 是否可调用、norm/RoPE 次序与舍入、bias、GQA、输出布局、UB/tiling；不能由名称推断 |
| Norm + RoPE + cache 写入 | vLLM-Ascend Triton 的模型专用 QK norm/rope/cache 复合实现 | paged cache 与 DiT prefix cache 区别、slot mapping、写入生命周期、辅助 index 分支、已旋转缓存不能重复处理 |
| Linear + activation / 完整 FFN | 部署版本的 GEMM epilogue、MindIE-SD `eagle_ffn_linear`、torch_npu `npu_ffn` 及 Ascend Triton 实现 | packed gate/up 权重顺序、GEMM 输出与激活舍入、权重预处理/额外常驻内存、TP 归约和量化限制 |
| SDPA / QK-softmax-V | MindIE-SD `attention_forward`、torch_npu FA/FIA 系列 | causal/mask、GQA/MQA、真实 seq len、cache、LSE、layout |
| MoE 分组 GEMM + activation | `npu_grouped_matmul` 及目标版复合算子 | EP dispatch/combine、routing、权重布局、padding、量化 scale |

这些是检索种子，不保证目标版本有此符号或适配目标模型。量化、稀疏、近似 attention 独立列为改变精度/算法的候选，仅在用户允许时进一步分析。

按 [复合融合检索](prior-art-and-compound-fusion.md#复合融合检索) 对比更大边界。公共 API、私有 wrapper、schema/注册、kernel、调用方和测试分别核对；“仓内有源码”与“部署包已编译/注册且被消费”分别记录。复合算子改变内部精度边界时，即使输入/输出都为 BF16，也不能认定与分解路径等价。

## 每个候选的契约记录

```text
operator/provider/package version/CANN/hardware:
official URL + doc version + accessed date; installed schema/source tag:
model symbol + branch + mathematical expression and operation order:
input/output shape, stride, layout, dtype, device, alignment:
weights/bias/eps, accumulation and cast order:
mask/causal/seq lengths/Nq:Nkv/rotary convention:
tuple order, auxiliary outputs, alias/in-place and all consumers:
workspace, weight preprocessing, cache/stream/graph lifetime:
adaptation transforms and their cost:
matched / conflicting / unverified constraints:
capability gate, original fallback, precision and profile validation:
```

API 存在性、形状可调用、数学等价、实际消费和测得收益是不同证据。复合算子不兼容时，考虑较小融合边界，而不是交换运算顺序或伪装布局使其可调用。
