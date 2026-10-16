# Profiling 解析与源码归因

## 选择输入

先清点文件，不递归载入所有 trace。选择同一请求/场景代表性的 rank/stage；仅在发现不平衡或通信瓶颈后扩展。保留原始文件，不改写采集结果。确认 warmup、编译、初始化、请求区间是否在 trace 内。

- `trace_view.json`、Chrome `trace.json`、`.json.gz`：先检查 metadata 中 process/thread 名称、category、时间单位，分清 host op、runtime、device kernel、HCCL、memory copy、用户标记及 summary。
- `kernel_details.csv`、`op_summary*.csv` 等：检查表头与导出版本，确认行是一次执行还是聚合。脚本只接受一次执行耗时行；聚合表要使用其 Count/Total/Avg 语义单独分析，不把一行当一次调用。
- `operator_details.csv` 与关联表：可提供 host/device 映射；以实际列名和 ID 关系为准。
- XLSX：选择相应工作表导出 CSV，记录转换工具、sheet、列及单位。不能只看文件名判断数据含义。
- 原始二进制/数据库：用目标环境对应 torch_npu/msprof 官方离线导出流程。读取版本文档后给出准确命令；不要对未知 schema 猜 SQL。

仓内 `vllm_omni/platforms/npu/profiler.py` 可核对采集方式。当前示例实现使用 torch_npu profiler，offline analyse 与普通 torch profiler 的 key_averages 不同；参数和导出格式仍以目标版本为准。

## 脚本使用

脚本为 Python 标准库实现，无 torch/NPU 依赖。路径相对于 skill 目录；在分析报告中保留实际命令。

```sh
python scripts/summarize_profile.py trace trace_view.json --inventory
python scripts/summarize_profile.py trace trace_view.json --category '<observed device category>' --pid '<observed device pid>' --unit us --output profile-summary.json
python scripts/summarize_profile.py csv kernel_details.csv --inventory
python scripts/summarize_profile.py csv kernel_details.csv --name-column 'Name' --duration-column 'Duration(us)' --shape-column 'Input Shapes' --unit us --output kernel-summary.json
```

示例列名、category、pid 都必须替换成实际值。同文件包含多个设备时分别运行，按 metadata 验证 pid 确实属于该设备。可用 `--tid` 进一步限制轨道。CSV 混合设备时先按实际 Device ID/Rank 列过滤并保留过滤依据。

Trace 只处理 `ph=X` 的完整区间事件；B/E 配对事件需要另行转换。存在 B/E 时脚本披露数量，避免静默假装完整。没有匹配事件时报错。默认不猜 NPU category、不按算子名自动区分 host/device，也不把 `displayTimeUnit` 当作 timestamp 单位。

输出按 `(name, shape)` 聚合 count/sum/mean/max ms，保留样本行号或事件下标、args 和区间。Trace 同时输出所选事件区间并集及观测跨度；这些不是请求耗时或设备利用率。脚本不连接 flow、不排除 HCCL、不计算 self time，也不估计端到端收益。

## 避免时间误算

同一 kernel 的 host wrapper 和 device event 不能相加。嵌套 annotation/算子 inclusive time 不能当 self time。多 stream、多 rank 并发时间不能相加当墙钟时长；kernel sum 可能超过观测跨度。HCCL 等待和计算可能重叠；区间并集不是融合可节省时间。

记录热点覆盖的请求/层/步数，不能把采样几层直接外推到完整请求。无 shape、stack 或 correlation ID 时，写清当前归因不足，提出最小补采范围。不要为了分析已有 trace 先强制跑一轮完整模型。

## 建立证据链

1. 在热点事件 args/关联 CSV 中找 correlation/flow、输入 shape、module/stack。
2. 映射到真实 forward 分支及消费者，检查 dispatch 与实际包实现。
3. 对照生产参数确认重复次数、shape 和调用序列一致。
4. 若只能名称匹配，标为低置信度候选；补采模块标记/调用栈后才能称热点归因。

收益验证用不启用 profiler 的基线与 A/B，固定输入、seed、steps、cache 状态、编译设置、并行拓扑和 warmup。至少给出重复次数、median 与波动范围；慢路径 fallback 也应单独记录。
