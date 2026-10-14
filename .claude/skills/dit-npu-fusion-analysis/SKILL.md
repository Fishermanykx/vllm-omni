---
name: dit-npu-fusion-analysis
description: Analyze NPU operator fusion opportunities for DiT image and video models such as HunyuanImage and Wan2.2 in vLLM Omni by joining existing profiling, active model call chains, and version-matched MindIE-SD or torch_npu contracts. Use for evidence-backed DiT fusion feasibility and implementation plans; code alone yields a preliminary audit, not measured speedup. Pure AR and VAE optimization are outside the primary scope.
---

# DiT NPU Fusion Analysis

针对 HunyuanImage（hyimg）、Wan2.2 等 DiT 类图像/视频模型，根据实际 profiling、模型执行路径和昇腾算子契约，生成可复核的融合机会报告。默认交付分析与适配设计；用户要求实现时，沿用报告中的契约、回退和验证方案。以当前源码和运行环境为准，不把相近模型的优化清单当作目标模型事实。报告使用用户语言。

聚焦 denoising transformer 中的 Attention、Norm、RoPE、调制、残差、MLP/MoE 与布局融合。对于混合 AR+DiT 模型，先识别生成图像/视频的 DiT 执行路径；AR 文本阶段和 VAE 用于划分端到端耗时边界，纯 AR 或 VAE 优化应转交对应流程。

## 输入与分析范围

优先从已有材料获取：目标仓及 commit/dirty diff、模型/checkpoint、启动和请求命令、硬件型号、torch/torch_npu/CANN/MindIE-SD/vLLM-Ascend 版本、profiling 路径、目标指标和精度要求。

记录 resolution、batch、token/head shape、denoise steps、CFG、TP/SP/CP/EP、compile/graph、量化与 cache 设置。分开 AR prefill/decode、DiT step0/step1+、VAE 和不同 rank。不能把 HunyuanImage 3.5 的 shape、算子 ABI 或拓扑直接用于 3.0。

缺少材料时继续独立工作：无 profile 做源码预审；无设备版本做条件性契约分析；只询问影响结论的缺项。不得虚构热点、设备探测、算子可用性或跑分。分析本身不要求启动服务或运行多卡推理。

## 工作流

### 1. 固定证据

记录源码快照、profile 文件及采集配置、rank/stage/请求边界和官方文档版本。已有非 profiler benchmark 作为性能基线；诊断 trace 的时间不作为最终性能指标。

读 [profiling 方法](references/profiling.md) 选择解析路径。辅助脚本支持 Chrome trace JSON/GZ 和显式列映射的逐次算子 CSV；先查看类别/轨道/表头再选择数据。脚本仅汇总事件，不自动判断融合或归因源码。

### 2. 从热点恢复执行子图

按 device 时间、调用次数和 shape 分组；将耗时归于实际 request/stage/step/rank。用 correlation/flow ID、stack、module 标记连接 host op 和 device kernel，标记连接的置信度。名称相似和时间相邻只能作为线索。

对每个热点沿 pipeline → model.forward → layer → CustomOp/平台 dispatch → vendor wrapper → 实际 kernel 追踪。核对分支条件、用户是否启用、shape/stride/dtype、参数与中间值消费者。搜索结果不是“路径已执行”的证据。

优先检查残差+Norm、Q/K Norm/RoPE、门控激活、投影布局、Attention 和 MoE；只有目标代码含有对应计算时才检查 AdaLN/GroupNorm。若瓶颈是 host gap、HCCL 或 VAE，应说明融合能覆盖的范围，不能强行归因到 elementwise。

### 3. 检索并核对算子契约

读 [算子与来源指南](references/operator-contracts.md)。检索目标版本的 MindIE-SD、昇腾社区 torch_npu 自定义 API、对应 tag 的 wrapper/schema/kernel/tests。文档、实际包签名和模型契约逐项对比；新版本网页不能证明旧部署包支持。

记录 API/命名空间、来源 URL/版本/日期、设备支持、数学公式及运算顺序、dtype/累积精度、shape/stride/layout、广播、eps、bias、输出 tuple 顺序、alias/in-place、副作用、workspace 和 graph 支持。未知项保留为待证，不能按经验补全。

从输入到所有消费者比较整个子图：例如残差输出必须保留、RoPE 的 half/interleaved/MRoPE 不能混用、norm 与 RoPE 不能交换次序、GQA 不能改成 MLA。布局转换、padding、cast、同步和数据搬运计入适配成本。

### 4. 分类与收益判断

每个候选同时给出三轴状态，避免把可适配写成已加速：

| 轴 | 状态 |
| --- | --- |
| 可行性 | `already-fused`、`direct`、`adapter`、`blocked`、`unknown` |
| 证据 | `source-only`、`profile-linked`、`npu-validated` |
| 收益 | `unmeasured`、`estimated`、`measured` |

`already-fused` 必须说明是源码已有调用还是已观察到 kernel；`direct` 表示契约匹配，仍不代表性能已验证。`blocked` 写清冲突及不改变模型语义的解法是否存在。列出不推荐的候选和原因。

优先级由热点贡献、可消除开销、适配成本、工程成本与精度风险共同决定。只有串行关键路径占比 f 与局部加速比 s 有依据时，才用 `1 / ((1-f) + f/s)` 作条件估算；不要用重叠 kernel 累加时间计算 f。共用子图的候选互斥或联合估算，不能叠加收益。

### 5. 给出可实施的适配与验证

指定目标符号、融合前后子图、adapter 位置、能力 gate、原始 fallback、参数/权重映射、生命周期和互斥条件。复用现有 CustomOp/平台 backend；已有等价融合优先复用。不要为了适配更改 checkpoint、全局 monkey patch 或扩大为量化/稀疏改造。

验证覆盖：生产 shape 的单算子对照 → 一层中间值 → 固定输入完整模型 → 关闭 profiler 的同配置 A/B。记录 atol/rtol 的来源、max abs/relative error、非有限值与必要的图像质量指标。精度阈值未明确时保持原精度，并将验收阈值标为待确认。测试非连续输入、边界/不支持 shape、分支和 fallback；有状态路径验证 cache/reset，分布式路径验证参与 rank。

从新 profile 证明目标 kernel 被消费、原 kernel/launch 减少且没有新增大量转换或同步。成功返回图像、包可 import 或 Python wrapper 被调用，都不足以证明融合生效。

### 6. 落盘交付

使用 [报告结构](references/report-template.md)，保存分析报告与原始汇总 JSON。每条推荐都能回溯 profile 事件/CSV 行、源码符号/行号和版本化文档。将事实、推断和待验证项分开。若用户只要求分析，交付报告和验证计划即完成，不自动修改模型。

分析 HunyuanImage 3.0 时，可读 [源码预审示例](references/hunyuan-image3-example.md)；重新核对当前 commit，示例不是热点清单或兼容性保证。
