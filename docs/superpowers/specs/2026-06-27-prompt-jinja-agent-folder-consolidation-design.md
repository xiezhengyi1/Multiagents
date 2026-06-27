# Prompt Jinja and Agent Folder Consolidation Design

> **Date:** 2026-06-27
> **Scope:** Conservative follow-up to the context engineering refactor.
> **Decision:** Preserve public agent package boundaries while improving prompt template reuse and merging small internal modules.

## Goal

Make `control_runtime.context.prompts` actually use Jinja2 as the prompt composition layer, then reduce clutter in `agents/grounding` and `agents/planning` without changing their public agent APIs.

## Non-Goals

- Do not remove `agents.grounding` or `agents.planning` packages.
- Do not redesign agent runtime loops, tool names, or response schemas.
- Do not move all compiler logic into `context` or `domain`.
- Do not preserve old context engineering entrypoints that were removed in the previous refactor.

## Prompt Template Design

`PromptEngine` becomes the only prompt rendering primitive for system prompts and structured user prompts. It should configure Jinja2 with:

- `StrictUndefined`, so missing template variables fail loudly.
- `trim_blocks=True` and `lstrip_blocks=True`, preserving readable plain-text prompt output.
- A JSON filter backed by `json.dumps(..., ensure_ascii=False, default=str)`.
- A small `clean` or `dedent` helper only if needed to keep rendered prompts stable.

The template tree keeps the existing top-level shape but fills in real reusable fragments:

- `templates/shared/base_system.j2` defines required blocks for role, responsibility, rules, tool discipline, knowledge search, and output contract.
- `templates/shared/macros.j2` provides small list/section helpers for consistent prompt text.
- `templates/shared/knowledge_search.j2` contains the shared 3GPP knowledge-tool discipline.
- `templates/shared/tool_discipline.j2` contains duplicate-call and allowed-tool rules.
- `templates/shared/output_contracts.j2` contains raw-JSON and schema-shape constraints.
- `templates/main/system.j2`, `templates/grounding/system.j2`, `templates/planning/system.j2`, and `templates/single/system.j2` extend the shared base and fill only agent-specific blocks.
- `templates/planning/user.j2` replaces Python string concatenation in `build_advisor_user_prompt`.

The existing exported constants stay available:

- `MAIN_CONTROL_SYSTEM_PROMPT`
- `IEA_SYSTEM_PROMPT`
- `OSA_SYSTEM_PROMPT`
- `SINGLE_AGENT_ROUND_PROMPT`
- dynamic rule constants

Those constants should be generated through builders/rendering rather than independent string concatenation. This keeps existing callers stable while making Jinja2 the source of truth.

## Builder Design

`PromptBuilder` remains the common base. Concrete builders should render templates directly:

- `MainPromptBuilder.system_prompt()` renders `main/system.j2`.
- `GroundingPromptBuilder.system_prompt()` renders `grounding/system.j2`.
- `PlanningPromptBuilder.system_prompt()` renders `planning/system.j2`.
- `PlanningPromptBuilder.advisor_user_prompt()` renders `planning/user.j2`.
- `SinglePromptBuilder.system_prompt()` renders `single/system.j2`.

The Python prompt modules remain as compatibility facades. They can hold small dictionaries of prompt text fragments, examples, and dynamic-rule strings, but they should not contain large hand-concatenated final prompts when a Jinja template can express the same composition.

## Grounding Folder Consolidation

Keep `agents/grounding/agent.py`, `compiler.py`, `contracts.py`, `artifact_compiler.py`, `validator.py`, and `tool_result_adapter.py` as separate files because each has a clear reason to exist.

Merge small helper modules into `agents/grounding/common.py`:

- `directives.py` becomes `MainDirectiveExtractor` inside `common.py`.
- `qos_envelope_builder.py` becomes `QosEnvelopeBuilder` inside `common.py`.

After consolidation, internal imports should target `agents.grounding.common`. Compatibility shim files are not kept unless a tracked external import requires one; the previous user instruction was to delete residual old code, so the default is deletion plus test coverage.

## Planning Folder Consolidation

Keep `agents/planning/agent.py`, `compiler.py`, `planning_artifact_compiler.py`, `planning_validation.py`, `response_models.py`, `tools.py`, and `tool_result_adapter.py`.

Merge `policy_normalizer.py` into `planning_artifact_compiler.py` because the normalizer is only a planning artifact concern and is already heavily coupled to policy draft normalization. `planning_validation.py`, `tools.py`, and `agent.py` should import normalization helpers from `planning_artifact_compiler.py`.

Do not merge `tool_result_adapter.py` into `agent.py`; it is a runtime-boundary parser and keeping it separate prevents `agent.py` from growing further.

## Compatibility Contract

These imports must continue to work:

- `from control_runtime.context.prompts import GroundingPromptBuilder, PlanningPromptBuilder, RetryPromptBuilder`
- `from control_runtime.context.prompts import IEA_SYSTEM_PROMPT, OSA_SYSTEM_PROMPT, MAIN_CONTROL_SYSTEM_PROMPT, SINGLE_AGENT_ROUND_PROMPT`
- `from control_runtime.agents.grounding import IntentEncodingAgent`
- `from control_runtime.agents.planning import OptimizationStrategyAgent, OptimizationStrategyCompiler`

These imports should no longer be used after consolidation:

- `control_runtime.agents.grounding.directives`
- `control_runtime.agents.grounding.qos_envelope_builder`
- `control_runtime.agents.planning.policy_normalizer`

## Testing Strategy

Use TDD for each behavior change.

Prompt tests:

- Rendering each system prompt through its builder includes agent-specific role text and shared output-contract text.
- Rendering fails when a required template variable is missing.
- Compatibility constants equal builder-rendered system prompts.
- `PlanningPromptBuilder.advisor_user_prompt()` renders JSON evidence and callable-tool policy via `planning/user.j2`.

Consolidation tests:

- Removed helper modules no longer exist.
- No tracked code imports removed helper modules.
- Public package imports still resolve.
- Existing contract tests for grounding, planning, single-agent prompts, and multi-agent protocol still pass.

Verification commands:

- `python -m pytest -q tests/test_context_engineering_refactor.py`
- `python -m pytest -q`
- `rg` scan for removed module imports under `src` and `tests`.

## Risks and Mitigations

- **Prompt text drift:** Tests should assert required clauses rather than exact full prompt snapshots, except where constant-builder equality is the compatibility requirement.
- **Circular imports:** Builders should accept fragment dictionaries or import prompt fragments lazily, matching the current facade style.
- **Over-merging:** Keep large files separate. Consolidate only modules that are small and directly coupled to one internal consumer.
- **Dirty worktree:** Stage and commit only the files touched by this work. Existing unrelated changes remain untouched.
