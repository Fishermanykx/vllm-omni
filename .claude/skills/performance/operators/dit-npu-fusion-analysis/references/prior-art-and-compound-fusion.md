# 已有工作与复合融合检索

在提出优化方案、编写实验补丁或实现候选前阅读。目的有两个：识别已有工作及本次真正的增量；让局部融合候选继续与更大的、契约相容的子图比较。

## 已有工作查重

先固定目标 checkout 的 commit、dirty diff、部署包来源；分别查询本地已实现代码与远端已提出的工作。检索目标仓，涉及 provider 改造时再扩展至 MindIE-SD、vLLM-Ascend/op-plugin 等相关仓，范围围绕已识别热点，不做无界仓库普查。

用可用的 GitHub connector、`gh` 或网页搜索做只读查询。例如：

```text
repo:vllm-project/vllm-omni is:pr "Qwen-Image"
repo:vllm-project/vllm-omni is:pr "shared modulation"
repo:vllm-project/vllm-ascend is:pr "norm rope"
```

交叉使用模型别名（QwenImage21/Qwen-Image-2.1、Wan/Wan2.2、HunyuanImage/hyimg）、子图（conditioning/modulation、QK norm/RoPE、SwiGLU/FFN）、实际符号与文件路径。标题/正文搜索不能覆盖全部 diff；对命中的相关 PR 读取文件列表和当前 head diff，而不是只读标题或最早描述。候选头变化或实施范围扩大时刷新。

查询覆盖 open、merged、closed，关联 issue、后继/替代 PR；closed-without-merge 的精度失败或性能回退也可能直接影响决策。每条相关记录包括：

- URL、状态、查询日期、head SHA、base、变更符号/子图；无法取得的字段注明未知。
- 平台、版本、shape、默认/实验开关、graph/parallel/cache 限制、测试与作者报告的结果边界。
- 本地分支是否包含同一实现，是否被条件分支挡住，已安装包是否来自其他 revision。
- 关系：已有实现 / 已有提案覆盖 / 需要平台适配 / 明确新增缺口；选择复用、验证、补缺口或不适用及理由。

检查 commit ancestry 可辅助判断是否合入，但 squash、cherry-pick、回退和 dirty patch 仍需对照实际代码。PR 状态与优化可用性分开：merged 不证明当前设备路径执行；open 不意味着方案不存在；本地 CUDA-only gate 不证明 PR 最新 head 同样未覆盖 NPU。

如果已有 PR 覆盖目标算法，优先验证或适配其实现，并明确本次新增的是平台支持、契约修正或回归证据。不要重做同等方案再标成新优化。复用 PR 不等于直接安装/切换到其分支，仍遵守用户范围和现有实验隔离方式。搜索失败/不完整时写明未知，不以空搜索结果证明无重复工作；可继续源码预审。

### Qwen-Image-2.1 示例

[Omni PR #8729](https://github.com/vllm-project/vllm-omni/pull/8729) 是检索 shared conditioning/modulation 的实际种子。2026-10-15 读取的页面也描述了 NPU interleaved RoPE 和后续实验。使用时重新读当前 head，比较其实现、开关与目标 checkout；共享 modulation 或 RoPE 的 NPU 验证应归属于相应已有工作，不将本地复现包装为新方案。其原环境/工作负载的性能不能作为当前请求的收益。

## 复合融合检索

### 从真实子图扩展边界

从 profile-linked 的活跃算子开始恢复生产者和所有消费者，再寻找更大覆盖：

| 起点 | 进一步检查的边界 |
| --- | --- |
| RoPE | QK Norm+RoPE → packed QKV split+Norm+RoPE → 加输出 layout 或 cache 写入 |
| Norm/scale | token 选择+调制 → Norm+scale/shift → gated residual+Norm；保留 residual 输出 |
| SiLU×up | 双输入激活融合 → packed gate/up projection+激活 epilogue → 完整 FFN |
| Attention | Q/K/V 布局、padding、prefix concat 与 mask 的联合边界；区分后端切换和算子融合 |

共享调制前移、缓存不变量等重复计算消除单独归因；它们可能为融合创造条件，但不是新的 vendor 融合算子。

比较较大融合与局部候选，最终可选局部方案。若更大算子不可用，报告已查到的具体实现、契约冲突和较小 fallback，不在找到第一个可调用 API 时结束调查。

### 分层查找实现

1. **MindIE-SD**：公共 `layers` API → `_custom_ops`/fake schema → plugin 注册/ACLNN 调用 → `csrc/ops` 的 host tiling/kernel/tests。例如搜索 `norm_rope_concat`、`eagle_ffn_linear`、AdaLN/gated residual；不能因未从 `mindiesd.__init__` 导出就断言不存在。
2. **vLLM-Ascend**：`ops/triton/`（包括 `linearnorm/` 及模型子目录）、CustomOp schema/注册、Python wrapper、实际模型调用和 tests。检索 split/packed/fused/norm/rope/cache、epilogue/ffn 等组合；不局限于单个文件名或顶层 API。
3. **torch_npu/op-plugin/CANN**：确认安装版本中的 schema、底层入口与设备支持，识别前两者是否只是同一后端的包装。相同内核的多个入口不当作独立性能方案。

没有编译、注册、设备调用或 dispatch 证据时，仍只是源码候选。main、目标 branch、部署 wheel 的版本分别记录；不把新 main 的能力自动赋给旧包。

### 具体检索种子与约束

这些是实际源码中发现的实现，不是目标模型支持保证：

- MindIE-SD `norm_rope_concat`：此次环境源码 `f6979d68` 的 `RopeType` 包含 INTERLEAVE；检查 host 对 cos/sin dtype 的限制、Norm 后是否有 BF16 舍入、输出 BNSD 布局、AutoContiguous 和 encoder/cache 语义。不要因它融合更多就忽略精度边界，也不要仅凭某版本冲突排除其他 Norm+RoPE 实现。
- [Ascend split QKV+RMSNorm+RoPE](https://github.com/vllm-project/vllm-ascend/blob/7260f67cf582874a646eb3cbf37a78848b272467/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py)：本地该 revision 可见 packed 输入、positions/cos_sin_cache、Q/K weights、Q/K/V 输出及 UB tiling。不能直接把 DiT 的 complex frequency tensor当作它的 position cache；核对 rotary 配对、输出 stride、权重与舍入。继续查同目录 TP、SIMT、MRoPE、V-norm 变体；额外 V normalization 会改变模型，不能为了接口匹配而启用。
- [Ascend QK Norm+RoPE+cache](https://github.com/vllm-project/vllm-ascend/blob/7260f67cf582874a646eb3cbf37a78848b272467/vllm_ascend/ops/triton/msa/minimax_qknorm_rope_cache.py)：该实现带 main/index 分支、slot mapping 与 paged cache 写入。可作为大融合设计参考，但这些状态契约不同于 DiT prefix KV cache，不能仅因 head_dim=128 就直接调用。

逐项重新确认目标 tag、真实注册名、wrapper、调用方与测试。路径/符号已变更时重新搜索，不把种子维护成“支持算子大全”。

### 决策与停止边界

为每个主要热点形成“原始子图 → 局部融合 → 较大融合”的比较记录：覆盖节点与所有输出、注册/部署可用性、数学与 cast 顺序、layout/stride/broadcast、缓存写读/alias、TP/SP 通信、UB/workspace、graph 与权重生命周期，以及新增 copy/cast/padding 的成本。

发现以下冲突时可以停止该具体实现的适配，保留相容的小边界，并说明是否有可行的改造路线：

- Norm 与 RoPE 之间原有舍入被消掉、cos/sin 精度不匹配或旋转约定不同。
- 某输出仍有外部消费者，却未返回；提前 cache 写入改变了生命周期或重复处理已经 Norm/RoPE 的 K。
- V-norm、index/MoE/量化/稀疏计算是额外语义，不能关闭或不在授权范围。
- 额外转换/权重常驻内存/通信成本抵消收益，或设备/UB/版本不满足。

比较原始后端、局部与复合候选时使用同一 reference、shape 和精度门槛。互相包含的融合分别单开测量，再测组合，不叠加覆盖同一子图的收益。即便某子算子已经融合，跨算子的中间 tensor、layout 和 launch 仍可能有增量价值，须继续检查。
