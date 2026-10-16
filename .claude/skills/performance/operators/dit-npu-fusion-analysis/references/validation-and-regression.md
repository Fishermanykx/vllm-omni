# 融合、后端与回归验证

## 可比较的实验

先锁定 checkpoint/revision、prompt/媒体及 hash、seed、图像尺寸/视频帧数、scheduler/sigmas/steps/guidance、dtype、cache/reset、拓扑、graph/compile 和有效后端。混合 AR+DiT 模型做融合隔离实验时固定 DiT 的文本条件或 AR 输出；另外保留真实端到端运行，区分两种测量范围。

在生产 shape 上建立小型矩阵，每次比较只回答一个问题：

| 配置 | 后端 | 融合 | 用途 |
| --- | --- | --- | --- |
| A | 当前有效实现 | 原状态 | 可复现基线 |
| B | 同一后端 | 指定新增融合 | 隔离融合增量 |
| C（适用时） | 契约匹配的现成候选 | 与 A 可比 | 后端选择对照 |
| D（适用时） | 候选后端 | 指定融合 | 已验证单项后的组合验收 |

融合与后端无法独立开关时，注明耦合，报告整体替换收益；不能伪称隔离实验。先淘汰不兼容硬件、dtype、shape/layout、mask、GQA、并行或 graph 的组合，再测量。Attention/Linear 的最佳实现是特定场景结果，不能泛化为全局最快。

## 精度参考与阈值校准

选择目标模型同版本的可信原始实现或仓内已验证基线，记录权重映射、输入预处理及输出解码约定。先跑 reference 自身重复实验，确认自运行方差；若 baseline/reference 已经不对齐，先解释差异，再归因融合。

单算子和一层记录输出及保留 residual/cache 等辅助状态，检查非有限值、max abs/relative error 与适用的 atol/rtol；接近零的分母需定义稳定处理。端到端图片可用 PSNR/SSIM 加误差分位数，视频增加逐帧/跨帧一致性和可复核片段。明确比较色彩空间、尺寸、帧对齐及编码是否影响指标。

用代表性 prompt/媒体、生产与边界 shape、允许的 dtype/拓扑校准阈值，记录样本、reference 自方差和质量要求依据。优先沿用已适用于同场景的仓内阈值；不能复制其他硬件阈值，也不能为使失败候选通过而降低阈值。质量标准尚未确定时可提出校准方案，不能宣称通过。

## 性能与生效证据

先单算子/一层对照，再做同配置固定工作量的非 profiler A/B；各配置使用相同 warmup，短名单至少三次有效测量，保存每次结果，报告 median/mean 与范围。少量样本不声称 p95/p99；吞吐比较保持相同 concurrency/arrival policy。

同时报告局部 kernel/launch、denoise stage、服务端生成时间、客户端 E2E、throughput、各 rank 峰值内存和失败请求。热身、编译和测量开销单独记录；profiling latency 仅用于诊断。把统计不足、方差覆盖提升、内存变差或 fallback 触发写成验收限制。

用独立诊断 trace 核实有效融合 kernel，核对输入适配、launch/intermediate traffic 与新增转换/同步。内存流量若没有 counters 或算子读写依据，只能写估算。实现目标是降低开销且满足质量约束；局部收益未改变 E2E 时保留局部结论，不宣称端到端加速。

## 复用当前仓库的回归入口

以下链接相对 skill 的 references 目录；个人安装时从目标仓读取。先检查当前源码、fixture、CLI 和 schema，再改配置：

- [accuracy tests](../../../../../../tests/e2e/accuracy/)：选择相同模型和任务已有测试；[HunyuanImage pixel accuracy](../../../../../../tests/e2e/accuracy/test_hunyuan_image3_pixel_accuracy.py) 可参考 reference/阈值组织，不直接复制其设备阈值。Wan 视频使用目标仓相应视频比较方法。
- [perf runner](../../../../../../tests/dfx/perf/scripts/run_benchmark.py) 与 [benchmark configs](../../../../../../tests/dfx/perf/tests/)：复用当前参数/marker/结果格式；[Wan2.2 配置](../../../../../../tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json) 含不同硬件场景，各条目的支持情况必须分别检查。
- [NPU ready CI](../../../../../../.buildkite/npu/test-npu-ready.yml) 与 [NPU nightly CI](../../../../../../.buildkite/npu/test-npu-nightly.yml)：检查真实硬件队列、卡数、依赖过滤和 artifacts；CUDA job 通过不等于 NPU 已覆盖。

单算子 CPU 可检验的索引/gate/fallback 逻辑与 NPU 数值测试分开；模型回归复用仓内 fixture/marker。实现并接入 CI 被用户要求时，在相关现有测试/config 中覆盖融合开/关、支持与不支持 shape、必要的 compile/cache/并行组合；选择风险相关组合，不穷举无关笛卡尔积。

性能配置记录硬件型号、版本、工作负载、重复测量和原始结果路径；基线数字必须来自实际目标环境。定义 latency/throughput/memory 的回归方向及阈值依据，不以新结果覆盖旧基线掩盖回退。目标设备资源不足时，交付可审查的配置变更及未执行原因，不能添加无依据的支持标记或假基线。

在对应 NPU 环境核对参数后，可复用当前 nightly runner 形式：

```sh
pytest -s -v tests/dfx/perf/scripts/run_benchmark.py -m npu --test-config-file <verified-config.json>
```

报告保存实际 accuracy/benchmark 命令、CI job/config 路径、基线 artifact、门槛和执行结果；未连到实际 pipeline 的新配置只能称“已准备”，不能称“已启用 CI”。仅分析任务交付接入方案即可，不运行完整多卡回归。
