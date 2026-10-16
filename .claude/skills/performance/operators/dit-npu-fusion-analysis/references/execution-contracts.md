# Versioned capability and execution contracts

These local contracts implement the composability suggestions in [RFC #8856](https://github.com/vllm-project/vllm-omni/issues/8856). They are not a repository-wide standard or an external RSI API. Version `1.0` identifies this skill's JSON format; bump the major version for incompatible field or semantic changes. The skill version identifies the instructions separately from the contract version.

## SkillMetadata

[skill-metadata.json](../skill-metadata.json) is the selection contract. Keep its `capability_id` stable across directory moves. Update `skill_version` when instructions change, and record the source revision or content hash in execution routes so two local edits with the same version remain distinguishable.

Required fields: `schema_version`, `capability_id`, `skill_version`, `name`, `entrypoint`, `capabilities`, `applicability`, `inputs`, `outputs`, `dependencies`, `expected_evidence`, `contracts`, `cost_hints`.

- Applicability includes workload/platform/mode, preconditions, degraded modes and exclusions. A missing profile permits source pre-audit, not a measured hotspot claim.
- Dependencies separate required resources, conditional runtime/tool requirements, and optional guides. Resolve repository guide paths relative to the canonical skill directory. For a personal installation, resolve optional guides from the target repository's `.claude/skills/`; their absence does not prevent standalone analysis.
- Cost hints are optional estimates. If supplied, each entry needs `value`, `unit`, `scope` and `basis`; do not invent numbers for unknown costs.

## ExecutionEvidence

Copy [the template](../assets/execution-evidence.template.json) to the task output directory as `execution-evidence.json`. Populate actual values; it is a format example, not an executed result. Unknown values remain JSON `null`, never zero or a fabricated success. Empty arrays mean no entries recorded, not that the corresponding checks passed. Use the user's language in explanatory values.

| Field | Type and contents |
| --- | --- |
| `schema_version` | String, `1.0` for this contract |
| `execution_id` | Nonempty string uniquely identifying this execution, e.g. a UUID |
| `prior_executions` | Array of `{execution_id, evidence_artifact}`; link prior runs without rewriting their results |
| `task_state` | Object with `mode` (`analyze` / `implement`), goal, authorized scope, and `before` / `after` snapshots |
| `skill_route` | Array of actually loaded capability IDs, skill versions, source revision/hash, entrypoint and routing reason; a planned guide is not an executed route |
| `evidence_reuse` | Array of `{execution_id, claim, applicability, decision, reason}`; decision is `reuse` / `revalidate` / `reject` |
| `changes` | Array of `{path, kind, diff_artifact, candidate_id}`; empty for analysis without code/config changes |
| `metrics` | Comparison claim, reference, controlled reference adaptations, threshold/calibration provenance and measurements |
| `cost` | Separate `measured` and `estimated` arrays; entries contain value, unit, scope and source/basis |
| `outcome` | Task status, optimization acceptance, production qualification and per-step outcomes |
| `artifacts` | Array of `{kind, path, description}` for actual reports, diffs, raw records, outputs or evidence |
| `limitations` | Array of strings explaining missing evidence, unsupported scope or inconclusive results |

Task snapshots identify repository/revision/dirty diff, model and checkpoint revision, configuration, workload and environment. Include shape/resolution/steps/seed/CFG, dtype, parallelism, graph/compile/cache, device and provider versions when relevant to the comparison. Record changes between snapshots even when the checkpoint is unchanged.

Each measurement contains `name`, `baseline`, `candidate`, `unit`, `scope`, `repeats`, `spread`, `status`, `artifact` and `provenance`. Record per-variant repeat counts and dispersion in objects when they differ. A delta is optional and needs a stated formula. Trace kernel sum and request latency are distinct scopes. A profile-derived estimate must be labeled as estimated rather than reported as measured E2E gain.

Task status is `complete` / `partial` / `blocked`. Optimization and production acceptance use `pass` / `fail` / `skipped` / `unvalidated` / `inconclusive`. Per-step and measurement statuses use `pass` / `fail` / `skipped` / `unvalidated` / `inconclusive`, with a reason for non-passing states. Analysis can be complete while optimization remains unvalidated. Use `inconclusive` when execution happened but cannot support a conclusion; use `unvalidated` when the claim has not been tested. Do not label an unmeasured result pass.

Per-step entries contain `name`, `status`, `reason` and `artifacts`. Quality claims distinguish aligned-reference parity from feature-on/off fidelity. Record reference changes, threshold source/calibration and controlled variables; neither numerical claim automatically establishes production readiness.

## Reuse and composition

Before reusing evidence, compare its revision/dirty changes, configuration, workload, environment, reference and threshold scope against the current task. Record why it applies, or what needs revalidation. Preserve the original execution ID/artifact; inherited measurements must retain their original provenance.

The execution agent owns coordination. Load only guides needed to fill evidence gaps, and attribute each candidate's changes and results separately. Routing history alone does not establish causality; performance claims require controlled comparisons. Return evidence to the caller or output directory; this contract does not authorize external publication, scheduling or autonomous self-improvement.
