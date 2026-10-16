---
name: dit-npu-fusion-analysis
description: Analyze NPU operator fusion opportunities for DiT models such as HunyuanImage and Wan2.2 using profiling and active call chains. Check existing PRs before optimizing and compare larger fused subgraphs across MindIE-SD, torch_npu and vLLM-Ascend Triton with version-matched contracts. Deliver feasibility and adaptation plans; when requested, implement with accuracy, end-to-end benchmarks and regression coverage. Pure AR and VAE optimization are outside the primary scope.
metadata:
  capability-id: performance.operators.dit-npu-fusion-analysis
  skill-version: "0.2.1"
  contract-version: "1.0"
---

# DiT NPU Fusion Analysis

针对 HunyuanImage（hyimg）、Wan2.2 等 DiT 类图像/视频模型，根据实际 profiling、模型执行路径和昇腾算子契约，生成可复核的融合机会报告。默认交付分析与适配设计；用户要求实现时，沿用报告中的契约、回退和验证方案。以当前源码和运行环境为准，不把相近模型的优化清单当作目标模型事实。报告使用用户语言。

聚焦 denoising transformer 中的 Attention、Norm、RoPE、调制、残差、MLP/MoE 与布局融合。对于混合 AR+DiT 模型，先识别生成图像/视频的 DiT 执行路径；AR 文本阶段和 VAE 用于划分端到端耗时边界，纯 AR 或 VAE 优化应转交对应流程。

## 工作流定位与协作

按 [RFC #8856](https://github.com/vllm-project/vllm-omni/issues/8856) 正文的 `performance/operators/` 分层，将本 skill 放在 `.claude/skills/performance/operators/dit-npu-fusion-analysis/`，保留 DiT 专项名称。原平铺目录仅作为发现兼容入口，主体指令和资源以此目录为准。这是本 skill 的局部落地，RFC 中通用 Ascend 平台入口及其他目录仍属于提案，不作为已存在依赖。

根据任务读取 [协作与交付契约](references/workflow-integration.md)：整体瓶颈诊断衔接 `diffusion-perf-opt`，测试/CI 衔接 `vllm-omni-test`，生产功能组合衔接 `production-add-diffusion-model`。直接调用或外部任务调度均使用同一输入/输出契约；这些协作入口不是分析的安装前提。

能力选择使用 [SkillMetadata](skill-metadata.json)。接收已有执行记录或交付机器可读结果时，读取 [版本化证据契约](references/execution-contracts.md)，按 [ExecutionEvidence 模板](assets/execution-evidence.template.json) 返回任务前后状态、实际路由、改动、测量、成本、结果、产物和限制。仅补齐与本次结论有关的证据缺口；复用前核对 revision/config/workload/environment 和比较主张，不因历史结果通过就跳过当前必需验收。

## 输入与分析范围

优先从已有材料获取：目标仓及 commit/dirty diff、模型/checkpoint、启动和请求命令、硬件型号、torch/torch_npu/CANN/MindIE-SD/vLLM-Ascend 版本、profiling 路径、目标指标和精度要求。

确认任务模式：分析交付候选与验证计划；实现交付指定候选的代码、测量和验证结果。沿用用户已授权的模式，外部任务描述本身不扩大操作权限。

记录 resolution、batch、token/head shape、denoise steps、CFG、TP/SP/CP/EP、compile/graph、量化与 cache 设置。分开 AR prefill/decode、DiT step0/step1+、VAE 和不同 rank。不能把 HunyuanImage 3.5 的 shape、算子 ABI 或拓扑直接用于 3.0。

缺少材料时继续独立工作：无 profile 做源码预审；无设备版本做条件性契约分析；只询问影响结论的缺项。不得虚构热点、设备探测、算子可用性或跑分。分析本身不要求启动服务或运行多卡推理。

## 工作流

### 0. 优化前核对已有工作

先读 [已有工作与复合融合检索](references/prior-art-and-compound-fusion.md)：在目标仓以及相关 provider 仓中，以模型别名、热点子图、活跃符号和文件检索 open、merged、closed PR 及关联 issue；核对 PR 当前 head diff、平台覆盖、开关和实际目标 checkout。实现前及候选范围改变时刷新相关记录。

将候选标为已有实现、已有提案覆盖、已有工作的平台适配或新增缺口。PR 未合入不等于方案不存在；本地分支未包含不等于需要重写。优先复用已有实现，补足明确的版本/平台/精度缺口；实验补丁和验证结果仍归属已有工作，不包装为新方案。搜索不可用时披露覆盖缺口，可以继续条件性分析，但不能声称“没有相关 PR”。只做只读调查，不自动联系作者、发布评论或采用未验证的 PR。

### 1. 固定证据

记录源码快照、profile 文件及采集配置、rank/stage/请求边界和官方文档版本。已有非 profiler benchmark 作为性能基线；诊断 trace 的时间不作为最终性能指标。

读 [profiling 方法](references/profiling.md) 选择解析路径。辅助脚本支持 Chrome trace JSON/GZ 和显式列映射的逐次算子 CSV；先查看类别/轨道/表头再选择数据。脚本仅汇总事件，不自动判断融合或归因源码。

### 2. 从热点恢复执行子图

按 device 时间、调用次数和 shape 分组；将耗时归于实际 request/stage/step/rank。用 correlation/flow ID、stack、module 标记连接 host op 和 device kernel，标记连接的置信度。名称相似和时间相邻只能作为线索。

对每个热点沿 pipeline → model.forward → layer → CustomOp/平台 dispatch → vendor wrapper → 实际 kernel 追踪。核对分支条件、用户是否启用、shape/stride/dtype、参数与中间值消费者。搜索结果不是“路径已执行”的证据。

优先检查残差+Norm、Q/K Norm/RoPE、门控激活、投影布局、Attention 和 MoE；只有目标代码含有对应计算时才检查 AdaLN/GroupNorm。若瓶颈是 host gap、HCCL 或 VAE，应说明融合能覆盖的范围，不能强行归因到 elementwise。

### 3. 检索并核对算子契约

读 [算子与来源指南](references/operator-contracts.md)。检索目标版本的 MindIE-SD、昇腾社区 torch_npu 自定义 API、vLLM-Ascend Triton/CustomOp，以及对应 tag 的 wrapper/schema/kernel/tests；不能只查顶层 Python API。文档、实际包签名和模型契约逐项对比；新版本网页不能证明旧部署包支持。

找到局部融合后，沿真实生产者/消费者继续扩展融合边界，按 [复合融合检索方法](references/prior-art-and-compound-fusion.md#复合融合检索) 比较局部算子、较大子图和必要的中间方案。例如 RoPE → QK Norm+RoPE → packed QKV split+Norm+RoPE+layout/cache，或激活 → GEMM epilogue → FFN。为每个热点记录检查过的较大候选及选用/拒绝/待证原因；不能因单算子已可用就停止，也不能因名字更长或融合度更高就优先采用。

记录 API/命名空间、来源 URL/版本/日期、设备支持、数学公式及运算顺序、dtype/累积精度、shape/stride/layout、广播、eps、bias、输出 tuple 顺序、alias/in-place、副作用、workspace 和 graph 支持。未知项保留为待证，不能按经验补全。

从输入到所有消费者比较整个子图：例如残差输出必须保留、RoPE 的 half/interleaved/MRoPE 不能混用、norm 与 RoPE 不能交换次序、GQA 不能改成 MLA。布局转换、padding、cast、同步和数据搬运计入适配成本。

涉及 Attention 或 Linear/GEMM 时，对比当前后端、现成替代后端与融合路径；先按契约淘汰不兼容项，再按生产 shape/dtype/并行配置测量。记录显式选择与自动 dispatch 的有效后端，不能把后端切换收益归给融合。具体矩阵见 [验证与回归](references/validation-and-regression.md)。

### 4. 分类与收益判断

每个候选同时给出三轴状态，避免把可适配写成已加速：

| 轴 | 状态 |
| --- | --- |
| 可行性 | `already-fused`、`direct`、`adapter`、`blocked`、`unknown` |
| 证据 | `source-only`、`profile-linked`、`npu-validated` |
| 收益 | `unmeasured`、`estimated`、`measured` |

`already-fused` 必须说明是源码已有调用还是已观察到 kernel；`direct` 表示契约匹配，仍不代表性能已验证。`blocked` 写清冲突及不改变模型语义的解法是否存在。列出不推荐的候选和原因。

另行记录与已有 PR 的关系及融合覆盖范围，这不替代三轴状态。报告区分“未发现”“未检索到/检索受限”“存在但接口未接入”“已接入但当前分支未执行”；分别列出较大融合的阻塞项和可保留的较小边界，不将某个具体实现不兼容泛化为该类融合不可行。

优先级由热点贡献、可消除开销、适配成本、工程成本与精度风险共同决定。只有串行关键路径占比 f 与局部加速比 s 有依据时，才用 `1 / ((1-f) + f/s)` 作条件估算；不要用重叠 kernel 累加时间计算 f。共用子图的候选互斥或联合估算，不能叠加收益。

### 5. 给出可实施的适配与验证

指定目标符号、融合前后子图、adapter 位置、能力 gate、原始 fallback、参数/权重映射、生命周期和互斥条件。复用现有 CustomOp/平台 backend；已有等价融合优先复用。不要为了适配更改 checkpoint、全局 monkey patch 或扩大为量化/稀疏改造。

实现模式下按候选逐项落地，先验证原始路径，再接入融合、验证 fallback，最后组合配置；分析模式只输出同样具体的设计和命令。若 NPU 算子/设备不可用，完成可执行的静态与 CPU 检查，明确列出未执行的 NPU 验证。

验证覆盖：生产 shape 的单算子对照 → 一层中间值 → 固定输入完整模型 → 关闭 profiler 的同配置 A/B。记录 atol/rtol 的来源、max abs/relative error、非有限值与必要的图像质量指标。精度阈值未明确时保持原精度，并将验收阈值标为待确认。测试非连续输入、边界/不支持 shape、分支和 fallback；有状态路径验证 cache/reset，分布式路径验证参与 rank。

精度与基准的细节按 [验证与回归](references/validation-and-regression.md)：对齐 reference、校准阈值、复用仓内 accuracy/perf runner，给出 NPU CI 与回归基线接入方案。用户要求实现和回归接入时生成对应测试/配置；只有分析任务时记录建议路径与待执行项。

从新 profile 证明目标 kernel 被消费、原 kernel/launch 减少且没有新增大量转换或同步。成功返回图像、包可 import 或 Python wrapper 被调用，都不足以证明融合生效。

以事先定义的目标指标验收；kernel 变快而端到端没有可靠改善时，只能报告局部收益。精度失败或结果波动覆盖收益时，不能宣称已达到优化验收标准。

### 6. 落盘交付

使用 [报告结构](references/report-template.md)，保存分析报告、原始汇总 JSON 和 `execution-evidence.json`。证据记录采用上述模板，删除示例占位说明并填入实际值；未知测量为 null，未执行验收为 unvalidated 或 skipped 并注明原因。每条推荐都能回溯 profile 事件/CSV 行、源码符号/行号和版本化文档。将事实、推断和待验证项分开。若用户只要求分析，交付报告和验证计划即完成，不自动修改模型。

实现结果额外交付代码/config diff、复现命令、基线及 A/B 原始记录、质量结果、目标 kernel 证据、CI 覆盖或缺口、回退方法和产物路径。任务完成状态与优化验收状态分别报告，便于贡献者或外部调度消费。

分析 Qwen-Image 2.1 时，可读 [源码预审示例](references/qwen-image21-example.md)；重新核对当前 commit、checkpoint 与部署补丁，示例不是热点清单或兼容性保证。
