# 融合机会报告结构

按实际场景裁剪，保留能复核结论的字段。所有性能时间统一 ms，原始数据单位另外记录。

## 结论与范围

先说明哪些机会值得验证、哪些已有融合/不兼容、缺少哪些决定性材料。标明是源码预审、profile 归因分析还是 NPU 实测。不能把 `unknown` 汇总成“支持”。

声明任务模式 `analyze/implement`、任务完成状态、优化验收状态及后续协作入口；任务完成与优化验收分开，字段见 [协作契约](workflow-integration.md)。

## 基线与证据索引

记录 repo/commit/dirty diff、模型与 checkpoint、硬件/软件版本、输入 shape/steps/seed、并行/量化/cache/graph、目标指标/精度要求；现有启动/请求/benchmark/采集命令，不知道就写缺失。

Profile 索引保留文件、rank/stage/step、采样范围、事件单位、选择的 category/pid/tid 或 CSV 列/过滤条件。引用源码符号、当前行号、官方版本化 URL 和访问日期。本地路径使用相对分析工作区路径，便于报告迁移。

## 已有工作核对

| 仓库 / PR 或 issue | 状态与核对日期 / head SHA | 已覆盖子图与平台 / 开关 | 与目标 checkout 的差异 | 决策：复用 / 平台适配 / 补缺口 / 不适用 |
| --- | --- | --- | --- | --- |

记录检索词、仓库和 open/merged/closed 覆盖，披露无法读取 diff 或检索受限等缺口。已提出但未合入的方案同样进入查重表；候选引用对应记录，区分已有工作的复现验证与新实现。PR 跑分只作为原作者、原环境的报告，不作为本次收益。

## 热点表

| ID | 场景/rank/step | op/kernel + shape | count | kernel sum ms | 占比口径 | 源码归因/置信度 | 事件证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |

重叠 kernel sum 不代表墙钟时间；无关键路径分母时不填写端到端占比。

## 候选总表

| ID | 融合前子图 → 候选 API | 可行性 | 证据 | 收益状态 | 热点覆盖 | 适配成本/风险 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |

## 每项决策卡

- 已有实现/提案及本次增量归属；较大融合候选的检查结果。
- 代码位置与实际调用链、分支；profile 事件下标/CSV 行号、关联字段及置信度。
- 融合前后计算表达式与所有输出消费者。
- provider、API、版本化来源；完整契约的匹配/冲突/未知项。
- 为什么比已有实现有增量价值；若已融合，说明 wrapper 与 device kernel 的证据区别。
- 接入符号、adapter、gate、fallback、workspace/权重/cache 生命周期及候选互斥。
- 条件性收益估算的输入、公式、假设，不能估算就明确写 `unmeasured`。
- 验证输入、精度标准及其来源、正负路径、目标 kernel 识别方法、非 profiler A/B 方法。
- 最终建议与尚待验证项。

## 融合边界比较

| 热点 | 局部候选 → 较大候选 | provider / 版本 / 注册与调用证据 | 覆盖及所有输出消费者 | 精度 / 布局 / cache / 通信冲突 | 适配成本与净收益证据 | 决策及 fallback |
| --- | --- | --- | --- | --- | --- | --- |

每个主要热点说明是否检查 MindIE-SD 复合算子、vLLM-Ascend Triton/CustomOp；未覆盖的 provider 及原因写明。较大算子 blocked 时指向具体实现、版本和冲突，不推断其他实现也不可行。互相包含的候选标记互斥/联合测量，避免重复计算收益。

## 实验与回归交付（实现或实测时）

| 配置 | 有效后端/gate | 实际运行次数 | 局部/denoise/E2E ms | throughput/peak memory | 质量指标/阈值 | 原始记录 |
| --- | --- | --- | --- | --- | --- | --- |

说明融合和后端实验是否隔离；列出 reference 对齐、阈值校准、自运行方差、A/B 变化及波动。未测字段留为未执行，不能填零。给出 kernel 生效证据，区分局部收益和端到端收益。

记录代码/config diff、实际命令、accuracy 测试、perf 配置、CI job/硬件及 artifacts、回归阈值依据与回退方法。列明 passed/failed/not-run，未启用的 CI 配置和缺失设备分别说明。分析任务仅填写计划和建议路径，不伪造执行结果。

## 不推荐项与后续验证

记录冲突、非热点、已有等价融合、转换成本过高或资料不足的候选。给出最小补充材料/实验及顺序；若实现已被用户授权，则按独立候选逐项实施验证。

## 版本化执行证据

随报告交付 `execution-evidence.json`，使用 [ExecutionEvidence 契约](execution-contracts.md) 和 [模板](../assets/execution-evidence.template.json)。记录 schema/skill version、execution ID、prior executions、TaskState 前后状态、真实 SkillRoute、Changes、Metrics、Cost、Outcome、Artifacts、Limitations。

比较主张注明是 aligned-reference parity、feature-on/off fidelity 还是性能改善；reference 适配与阈值/校准来源不能省略。每项验证使用 pass/fail/skipped/unvalidated，和任务交付完成状态分开。历史证据复用记录兼容性判断；不把过去的测量写成本次实测，不把估计成本写成实测成本，也不把数值通过等同生产就绪。
