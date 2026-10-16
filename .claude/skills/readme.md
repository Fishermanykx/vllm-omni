# Repository Skills for vLLM-Omni

This directory contains repository-scale skills maintained for `vllm-omni`.
They capture repeatable workflows for common contributor and maintainer tasks
such as model integration, CI-aligned testing, performance optimization, and
pull request review.

## Directory Structure

Each skill lives in its own directory under `.claude/skills/`, either at the
root or within a capability layer. Following the organization proposed in
[RFC #8856](https://github.com/vllm-project/vllm-omni/issues/8856), the DiT NPU
operator skill now uses this layout; other existing skills keep their paths:

```text
.claude/skills/
├── readme.md
├── dit-npu-fusion-analysis/
│   └── SKILL.md                  # Flat discovery compatibility entry
├── performance/
│   └── operators/
│       └── dit-npu-fusion-analysis/
│           ├── SKILL.md          # Canonical instructions
│           ├── skill-metadata.json
│           ├── agents/openai.yaml
│           ├── references/
│           ├── assets/execution-evidence.template.json
│           └── scripts/
└── ...                          # Other existing skills, unchanged
```

A skill may include:

- `SKILL.md`: the main workflow and operating instructions
- `references/`: focused reference material used by the skill
- `scripts/`: small helper scripts used by the skill
- `skill-metadata.json`: versioned capability/applicability/dependency contract
- `assets/`: templates used in task output, including execution evidence

## Using the Skills

Coding agents that discover repository skills can invoke the relevant skill by
name. For other agents, point them to the skill's `SKILL.md` and ask them to
read it before changing code. Combine a domain skill with `vllm-omni-test` when
tests or CI coverage change, then use `precheck-pr` for the contributor's final
self-check.

## Available Skills

- [`add-diffusion-model`](add-diffusion-model/SKILL.md): guides integration of
  a new diffusion model into `vllm-omni`
- [`add-recipe`](add-recipe/SKILL.md): creates or updates in-repository model
  recipes with verified task/input/output contracts, concrete hardware,
  shared feature-guide links, exact commands, and qualification evidence
- [`add-tts-model`](add-tts-model/SKILL.md): covers integration of new TTS
  models and related serving workflows
- [`diffusion-perf-opt`](diffusion-perf-opt/SKILL.md): guides diffusion model
  performance optimization, including profiling traces, parallel strategies,
  stage timing analysis, and benchmark-driven tuning
- [`diffusion-accuracy-test`](diffusion-accuracy-test/SKILL.md): adds or extends
  image/video numerical accuracy tests with aligned references, scoped
  tolerances, and explicit Buildkite Accuracy coverage
- [`dit-npu-fusion-analysis`](performance/operators/dit-npu-fusion-analysis/SKILL.md): analyzes NPU
  operator fusion opportunities for DiT models such as HunyuanImage and Wan2.2
    using profiling, active call chains, existing-PR checks, and version-matched
    MindIE-SD/torch_npu/vLLM-Ascend Triton contracts; compares local and larger
    fused subgraphs, with feasibility reports and, when requested, implementation,
  backend comparisons, accuracy/E2E validation, and NPU regression integration;
  see its [workflow routing](performance/operators/dit-npu-fusion-analysis/references/workflow-integration.md)
  and [versioned evidence contract](performance/operators/dit-npu-fusion-analysis/references/execution-contracts.md)
- [`find-simplifications`](find-simplifications/SKILL.md): finds evidence-backed
  opportunities to remove or merge dead, duplicated, speculative,
  over-generalized, or unnecessarily defensive vLLM-Omni code
- [`precheck-pr`](precheck-pr/SKILL.md): self-checks a branch before creating a
  PR by validating title format, dead code, simplification opportunities,
  accuracy and performance claims, and merge readiness
- [`production-add-diffusion-model`](production-add-diffusion-model/SKILL.md):
  takes a working Day-0 diffusion integration through API/limit parity,
  feature-combination evidence, quantization/offload, performance, production
  serving, hardware recipes, and CI readiness
- [`quantization`](quantization/SKILL.md): guides quantization method selection,
  model integration, checkpoint loading, and quality/performance validation
  for vLLM-Omni
- [`review-pr`](review-pr/SKILL.md): provides a frozen-snapshot, contract-aware
  workflow for maintainers and reviewers; contributors should use
  `precheck-pr` for self-review
- [`vllm-omni-npu-model-runner-upgrade`](vllm-omni-npu-upgrade/SKILL.md):
  upgrades NPU model runners to align with the latest vllm-ascend
  `NPUModelRunner`
- [`vllm-omni-test`](vllm-omni-test/SKILL.md): guides generation and execution
  of CI-aligned tests (L1–L4), pytest marker selection (`core_model` /
  `advanced_model` / `full_model`, `omni` / `tts` / `diffusion`), Buildkite
  wiring (`test-ready.yml`, `test-merge.yml`, `test-nightly.yml`,
  `test-weekly.yml`), and copy-paste local plus CI-like `pytest` commands; see
  `references/test-routing.md` for level-to-command mapping

## Task-to-Capability Composition

Compose only the capabilities required by the task and its evidence gaps.
These entries are guides loaded by an execution agent, not independently
running services. RFC-proposed guides are usable only when present in the
target checkout.

| Task | Coordination / owner | Capability to compose | Completion evidence |
| --- | --- | --- | --- |
| Identify DiT NPU fusion opportunities | Direct caller or workload execution agent | [DiT NPU fusion](performance/operators/dit-npu-fusion-analysis/SKILL.md) | Prior-art audit, profile/source attribution, contracts, candidate and validation plans |
| Optimize an existing diffusion model | [diffusion-perf-opt](diffusion-perf-opt/SKILL.md) or task execution agent | DiT NPU operator guidance when applicable; [testing](vllm-omni-test/SKILL.md) for required regression | Runtime route, quality claim, controlled E2E A/B, fallback/regression results |
| Integrate a model | [add-diffusion-model](add-diffusion-model/SKILL.md) | Operator guidance after semantic/loading correctness, when needed | Integration evidence plus scoped operator validation; no repeated onboarding for existing models |
| Qualify production behavior | [production-add-diffusion-model](production-add-diffusion-model/SKILL.md) | Applicable numerical/performance evidence and testing | Independent API, lifecycle, reliability and feature-combination qualification |

The DiT operator skill returns versioned `ExecutionEvidence`. Reuse checks
revision/configuration/workload/environment and comparison scope; a prior
numerical pass does not establish production readiness.

## MiniMax H3 Creative Skills

The [portable H3 collection](../../.agents/skills/README.md) contains the prompt
writing skill and eight creative workflows for product ads, 3D shorts, paper art,
brand promos, music videos, co-op intros, and hand-drawn/live-action fusion.
Same-named directories here link to the canonical `.agents/skills/` copies, so
Claude Code and Codex use the same instructions and templates.

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
