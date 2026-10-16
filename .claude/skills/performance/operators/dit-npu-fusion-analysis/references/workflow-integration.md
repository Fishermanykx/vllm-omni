# 工作流协作与任务交付

## RFC 依据与当前落点

[RFC #8856](https://github.com/vllm-project/vllm-omni/issues/8856)，2026-10-16 读取：正文按 shared methods、workload 和 hardware operators 分层，评论建议独立组合能力，并用版本化 SkillMetadata / ExecutionEvidence 保留执行证据。

本 skill 的主体位于 `.claude/skills/performance/operators/dit-npu-fusion-analysis/`，承担 DiT 工作负载下的 Ascend 算子专项。保留原平铺路径作为轻量兼容入口，避免依赖客户端是否递归发现 skill；不要同时执行两个入口。不迁移其他现有 skill，不创建未实现的平台/accuracy/benchmark 占位入口。

RFC 的 `ascend-kernel-opt` 是通用硬件能力提案；本专项保留用户指定的 DiT 名称，不声称覆盖完整 NPU profiling、通信、调度和生产资格。需要这些能力时由任务执行者组合对应实际指南。

## 按实际任务路由

以下是当前仓内链接。个人安装副本应从目标仓 `.claude/skills/` 读取对应资源；资源不可用时按本 skill 的自包含方法继续，并记录缺项。只加载当前任务相关内容。

| 当前需求 | 协作入口 | 本 skill 负责的部分 |
| --- | --- | --- |
| stage/step 空泡、调度、通信或整体性能 | [diffusion-perf-opt](../../../../diffusion-perf-opt/SKILL.md) | 提供 NPU profile 选择与算子归因；将非算子瓶颈交回工作负载流程 |
| 接入新 DiT 模型 | [add-diffusion-model](../../../../add-diffusion-model/SKILL.md) | 在基本语义/加载/执行路径正确后审计融合，保留未融合基线 |
| 生产配置、并行/cache/compile 组合 | [production-add-diffusion-model](../../../../production-add-diffusion-model/SKILL.md) | 验证融合 gate 与这些配置的相容性；复用 [performance patterns](../../../../production-add-diffusion-model/references/performance-patterns.md) 的已有约定 |
| accuracy、benchmark 和 CI 接入 | [vllm-omni-test](../../../../vllm-omni-test/SKILL.md) 及 [test routing](../../../../vllm-omni-test/references/test-routing.md) | 按当前仓规则选择测试层级、marker、NPU job，产出算子与模型回归证据 |
| 改变量化精度/权重格式 | [quantization](../../../../quantization/SKILL.md) | 先确认用户允许的精度范围，再联合核对融合的量化 ABI |

涉及 NPU runner 升级时再读取 runner-upgrade 工作流；不要为一个融合候选默认升级运行时。RFC 提出的通用 profiling/fusion/backend 文档及专用 accuracy/benchmark skill，只有实际存在后才能作为可调用依赖。

## 任务输入与模式

直接调用和外部调度使用相同字段：

```text
mode: analyze | implement
repository + revision + dirty changes:
model/checkpoint + workload + fixed inputs:
hardware + software versions + topology:
profile artifacts + baseline/commands:
objective + quality/precision constraints:
requested candidates + allowed execution scope:
output directory:
```

字段可缺失，按入口的缺项规则降级分析。外部调度选择任务；skill 本身不创建新任务、不定时轮询，也不自行扩展到远程发布或消息发送。已有实现授权无需再次询问；新增昂贵或外部操作仍按会话权限处理。

## 完成标准与结果契约

| 模式/阶段 | 完成标准 |
| --- | --- |
| 分析任务 | 证据索引、三轴候选分类、契约冲突、适配方案、验证计划及关键缺项已交付；不要求凭空补齐硬件实测 |
| 实现任务 | 范围内代码/配置完成，实际执行与未执行的验证分开记录；未完成所需验收时明确说明剩余工作 |
| 优化验收 | 目标 kernel 实际消费、质量达标、非 profiler 的目标指标改善有依据、所需回归覆盖与回退有效 |

人类报告可附以下摘要；机器可读交付使用 [版本化证据契约](execution-contracts.md)，只写实际存在的路径和结果：

```text
task_status: complete | partial | blocked
acceptance: not-evaluated | passed | failed | inconclusive
revision / changed_files / effective_backend_and_gates:
candidate_statuses: feasibility + evidence + benefit
prior_art: search_coverage / PR_head_and_status / reuse_or_platform_gap
fusion_boundaries: providers_checked / local_vs_compound / blockers_and_fallback
measurements: baseline / candidate / delta / spread / raw_artifacts
quality: reference / calibrated_thresholds / metrics / outputs
validation: commands / passed / failed / not_run_with_reason
regression: test_config / baseline / CI_job_or_gap
rollback / report_path / remaining_actions:
```

此处 `task_status` 只是交付报告字段，不调用任何目标状态管理工具。分析可以完成而优化验收尚未评估；实现受设备限制时不能写作已通过 NPU 验收。数据缺失不是零耗时或零误差。

## 按证据缺口组合，而非固定流水线

| 任务 | 协调/所有权 | 本专项输出 | 完成所需证据 |
| --- | --- | --- | --- |
| 仅识别融合机会 | 本 skill，直接调用或执行者选择 | PR 查重、热点/子图、契约与适配计划 | 来源可复核；缺项明确；实测验收可 unvalidated |
| 优化现有 DiT 算子 | 执行者或 diffusion-perf-opt 组织 | 独立候选补丁、gate/fallback、运行时命中 | 对照精度、非 profiler E2E A/B、必要回归；不重复模型接入 |
| 新模型接入后专项审计 | add-diffusion-model 负责语义/加载，本专项负责算子 | 不改变模型契约的融合建议或已授权实现 | 复用匹配的接入证据，补算子资格与精度/性能缺口 |
| 生产资格评估 | production-add-diffusion-model | 可复用的数值/性能证据及其范围 | 生产流程独立验证 API、生命周期、可靠性及功能组合 |
| CI 回归接入 | vllm-omni-test 负责仓内测试规则 | NPU case/config、命令及原始 artifacts | 真实 job/硬件结果；未执行与已通过分开 |

若目标 checkout 已有独立 accuracy/benchmark 指南，按能力和版本选用；不存在时使用本 skill 的验证参考及现有测试流程并记录路由。不要仅凭 RFC 提案名称报告其已被执行。外部 RSI 可以保存执行历史；本 skill 返回 execution ID 和 prior execution 链接，不实现自动学习或自修改调度系统。
