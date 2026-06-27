# Prompt Jinja and Agent Folder Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce current Python code volume by making Jinja2 templates the prompt source of truth and consolidating small grounding/planning helper modules without changing public agent APIs.

**Architecture:** Keep `agents.grounding` and `agents.planning` packages intact. Move prompt composition into `control_runtime.context.prompts.templates`, keep Python prompt modules as slim compatibility facades, and merge only small internal helper modules that have direct single-package coupling.

**Tech Stack:** Python 3, Jinja2, pytest, PowerShell, existing `control_runtime` agent packages.

---

## File Structure

Modify:

- `tests/test_context_engineering_refactor.py` - add prompt-rendering and consolidation regression tests.
- `src/control_runtime/context/prompts/engine.py` - configure strict Jinja rendering and JSON filter.
- `src/control_runtime/context/prompts/builders/base.py` - add a common `render_template()` helper.
- `src/control_runtime/context/prompts/builders/main.py` - render `main/system.j2`.
- `src/control_runtime/context/prompts/builders/grounding.py` - render `grounding/system.j2`.
- `src/control_runtime/context/prompts/builders/planning.py` - render `planning/system.j2` and `planning/user.j2`.
- `src/control_runtime/context/prompts/builders/single.py` - render `single/system.j2`.
- `src/control_runtime/context/prompts/main.py` - slim to dynamic rules plus compatibility constants generated from Jinja.
- `src/control_runtime/context/prompts/grounding.py` - slim to dynamic rules plus compatibility constants generated from Jinja.
- `src/control_runtime/context/prompts/planning.py` - slim to dynamic rules, examples, retry helper, and compatibility constants generated from Jinja.
- `src/control_runtime/context/prompts/single.py` - slim to compatibility constant generated from Jinja.
- `src/control_runtime/context/prompts/templates/shared/base_system.j2` - define required reusable blocks.
- `src/control_runtime/context/prompts/templates/shared/macros.j2` - add reusable section/list helpers.
- `src/control_runtime/context/prompts/templates/shared/knowledge_search.j2` - render knowledge-search skill blocks.
- `src/control_runtime/context/prompts/templates/shared/tool_discipline.j2` - centralize duplicate-call and allowed-tool rules.
- `src/control_runtime/context/prompts/templates/shared/output_contracts.j2` - centralize raw JSON output constraints.
- `src/control_runtime/context/prompts/templates/main/system.j2` - hold the main system prompt content.
- `src/control_runtime/context/prompts/templates/grounding/system.j2` - hold the IEA system prompt content.
- `src/control_runtime/context/prompts/templates/planning/system.j2` - hold the OSA system prompt content.
- `src/control_runtime/context/prompts/templates/planning/user.j2` - hold OSA user prompt composition.
- `src/control_runtime/context/prompts/templates/single/system.j2` - hold the single-agent system prompt content.
- `src/control_runtime/agents/grounding/common.py` - absorb `MainDirectiveExtractor` and `QosEnvelopeBuilder`.
- `src/control_runtime/agents/grounding/compiler.py` - import `MainDirectiveExtractor` from `.common`.
- `src/control_runtime/agents/grounding/artifact_compiler.py` - import `QosEnvelopeBuilder` from `.common`.
- `src/control_runtime/agents/planning/planning_artifact_compiler.py` - absorb policy normalization helpers and export them.
- `src/control_runtime/agents/planning/planning_validation.py` - import `normalize_app_id` from `.planning_artifact_compiler`.
- `src/control_runtime/agents/planning/tools.py` - import `json_friendly` from `.planning_artifact_compiler`.
- `src/control_runtime/agents/planning/agent.py` - import `normalize_app_id` from `.planning_artifact_compiler`.
- `src/control_runtime/agents/single/agent.py` - import `normalize_policy_plan_draft` from `.planning_artifact_compiler`.

Delete:

- `src/control_runtime/agents/grounding/directives.py`
- `src/control_runtime/agents/grounding/qos_envelope_builder.py`
- `src/control_runtime/agents/planning/policy_normalizer.py`

Do not touch unrelated dirty files under `experiments/`, `src/environment/`, `training/`, or unrelated docs.

---

### Task 1: Add Failing Prompt and Consolidation Tests

**Files:**
- Modify: `tests/test_context_engineering_refactor.py`

- [ ] **Step 1: Add imports for prompt builders and Jinja error type**

Add these imports near the existing prompt imports:

```python
from jinja2 import UndefinedError

from control_runtime.context.prompts import (
    GroundingPromptBuilder,
    MAIN_CONTROL_SYSTEM_PROMPT,
    OSA_SYSTEM_PROMPT,
    PlanningPromptBuilder,
    SINGLE_AGENT_ROUND_PROMPT,
)
from control_runtime.context.prompts.engine import PromptEngine
```

Keep the existing multi-name `from control_runtime.context import (...)` import. If `MainPromptBuilder` is already imported through `control_runtime.context`, do not duplicate it in the new import block.

- [ ] **Step 2: Add removed helper modules to the deletion contract**

Add this constant below `LEGACY_CONTEXT_ENGINEERING_FILES`:

```python
CONSOLIDATED_AGENT_HELPER_FILES = [
    "src/control_runtime/agents/grounding/directives.py",
    "src/control_runtime/agents/grounding/qos_envelope_builder.py",
    "src/control_runtime/agents/planning/policy_normalizer.py",
]
```

Add these patterns to `LEGACY_IMPORT_PATTERNS`:

```python
"agents.grounding.directives",
"agents.grounding.qos_envelope_builder",
"agents.planning.policy_normalizer",
```

Update `test_legacy_context_engineering_entrypoints_are_removed()` so it checks both lists:

```python
    for legacy_path in [*LEGACY_CONTEXT_ENGINEERING_FILES, *CONSOLIDATED_AGENT_HELPER_FILES]:
        assert not (ROOT / legacy_path).exists(), legacy_path
```

- [ ] **Step 3: Add a strict Jinja rendering test**

Append this test:

```python
def test_prompt_engine_raises_for_missing_template_variables() -> None:
    with pytest.raises(UndefinedError):
        PromptEngine().render("grounding/system.j2")
```

This test must fail before implementation because the current Jinja environment silently renders missing variables.

- [ ] **Step 4: Add builder/template compatibility tests**

Append this test:

```python
def test_prompt_builders_render_jinja_templates_as_compatibility_constants() -> None:
    assert MainPromptBuilder().system_prompt() == MAIN_CONTROL_SYSTEM_PROMPT
    assert GroundingPromptBuilder().system_prompt() == __import__(
        "control_runtime.context.prompts",
        fromlist=["IEA_SYSTEM_PROMPT"],
    ).IEA_SYSTEM_PROMPT
    assert PlanningPromptBuilder().system_prompt() == OSA_SYSTEM_PROMPT
    assert __import__(
        "control_runtime.context.prompts",
        fromlist=["SinglePromptBuilder"],
    ).SinglePromptBuilder().system_prompt() == SINGLE_AGENT_ROUND_PROMPT

    for rendered in (
        MainPromptBuilder().system_prompt(),
        GroundingPromptBuilder().system_prompt(),
        PlanningPromptBuilder().system_prompt(),
        __import__("control_runtime.context.prompts", fromlist=["SinglePromptBuilder"]).SinglePromptBuilder().system_prompt(),
    ):
        assert "Return raw JSON only" in rendered or "Return JSON only" in rendered
        assert "{% block" not in rendered
        assert "{% include" not in rendered
```

This test can partially pass today, but it protects compatibility while this implementation moves prompt text into Jinja templates.

- [ ] **Step 5: Add a planning user template test**

Append this test:

```python
def test_planning_user_prompt_is_rendered_from_jinja_template() -> None:
    prompt = PlanningPromptBuilder().advisor_user_prompt(
        normalized_user_intent={"session_id": "s1", "app_id": "app_1"},
        coordination_context={"active_domains": ["qos"]},
        planning_evidence={"flows": [{"flow_id": "flow-1"}]},
        available_tool_names=["preview_qos_optimizer"],
    )

    engine_rendered = PromptEngine().render(
        "planning/user.j2",
        normalized_user_intent={"session_id": "s1", "app_id": "app_1"},
        coordination_context={"active_domains": ["qos"]},
        planning_evidence={"flows": [{"flow_id": "flow-1"}]},
        tool_policy="Callable tools in this round:\n- `preview_qos_optimizer`",
        dynamic_rules=__import__(
            "control_runtime.context.prompts",
            fromlist=["OSA_DYNAMIC_RULES"],
        ).OSA_DYNAMIC_RULES.strip(),
        output_format_rules=__import__(
            "control_runtime.context.prompts.planning",
            fromlist=["_OUTPUT_FORMAT_RULES"],
        )._OUTPUT_FORMAT_RULES.strip(),
    )

    assert prompt == engine_rendered
    assert '"session_id": "s1"' in prompt
    assert "Callable tools in this round:" in prompt
```

This must fail before implementation because `planning/user.j2` does not exist.

- [ ] **Step 6: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q tests\test_context_engineering_refactor.py
```

Expected: FAIL with at least:

- `DID NOT RAISE <class 'jinja2.exceptions.UndefinedError'>`
- `TemplateNotFound: planning/user.j2`
- missing consolidated helper file assertion if the file deletion test runs first.

Do not edit production code until these failures are observed.

---

### Task 2: Implement Strict Prompt Engine and Real Jinja Rendering

**Files:**
- Modify: `src/control_runtime/context/prompts/engine.py`
- Modify: `src/control_runtime/context/prompts/builders/base.py`
- Modify: `src/control_runtime/context/prompts/builders/main.py`
- Modify: `src/control_runtime/context/prompts/builders/grounding.py`
- Modify: `src/control_runtime/context/prompts/builders/planning.py`
- Modify: `src/control_runtime/context/prompts/builders/single.py`
- Modify: prompt modules and templates listed in File Structure.

- [ ] **Step 1: Make PromptEngine strict and JSON-aware**

Replace `engine.py` with:

```python
from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

from ..budget import TokenBudget


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class PromptEngine:
    """Jinja2 environment with package-local template loading."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("control_runtime.context.prompts", "templates"),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
            keep_trailing_newline=False,
        )
        self._env.filters["json_dumps"] = _json_dumps

    def render(self, template_name: str, **context: Any) -> str:
        template = self._env.get_template(template_name)
        return template.render(**context).strip()

    def render_with_budget(
        self,
        template_name: str,
        budget: TokenBudget,
        **context: Any,
    ) -> str:
        rendered = self.render(template_name, **context)
        if budget.estimate(rendered) <= budget.limit:
            return rendered
        return rendered[: max(0, budget.limit * 4)]
```

- [ ] **Step 2: Add common builder render helper**

Replace `builders/base.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..engine import PromptEngine


@dataclass
class PromptBuilder:
    engine: PromptEngine = field(default_factory=PromptEngine)

    def render_template(self, template_name: str, **context: Any) -> str:
        return self.engine.render(template_name, **context)

    def system_prompt(self) -> str:
        raise NotImplementedError
```

- [ ] **Step 3: Populate shared templates**

Set `templates/shared/base_system.j2` to:

```jinja
{% block agent_role required %}{% endblock %}

{% block responsibility required %}{% endblock %}

{% block domain_rules %}{% endblock %}
{% block grounding_rules %}{% endblock %}
{% block planning_rules %}{% endblock %}
{% block tool_discipline %}{% endblock %}
{% block knowledge_search %}{% endblock %}
{% block output_contract required %}{% endblock %}
```

Create `templates/shared/macros.j2`:

```jinja
{% macro section(title, body) -%}
{{ title }}:
{{ body.strip() }}
{%- endmacro %}
```

Create `templates/shared/knowledge_search.j2` by moving the text currently exported from `knowledge_search.py` into template blocks. Keep separate blocks for IEA and OSA:

```jinja
{% macro iea_knowledge_search() -%}
{{ iea_knowledge_search_skill.strip() }}
{%- endmacro %}

{% macro osa_knowledge_search() -%}
{{ osa_knowledge_search_skill.strip() }}
{%- endmacro %}
```

Create `templates/shared/output_contracts.j2`:

```jinja
{% macro raw_json_contract(schema_name) -%}
Output contract:
- Return raw JSON only.
- Return exactly one {{ schema_name }} JSON object.
- Do not wrap the answer in markdown fences or prose.
{%- endmacro %}
```

Create `templates/shared/tool_discipline.j2`:

```jinja
{% macro no_duplicate_tool_calls() -%}
Tool discipline:
- Never call a tool that is absent from the current runtime.
- Never call the same tool twice with the same effective arguments in one round.
- Stop tool use once current evidence is sufficient for a valid final JSON object.
{%- endmacro %}
```

- [ ] **Step 4: Move system prompt text from Python constants into system templates**

Move each existing long prompt body verbatim into its matching `system.j2`, preserving exact wording where practical:

- `MAIN_CONTROL_CORE_PROMPT` body -> `templates/main/system.j2`
- `IEA_CORE_PROMPT` body -> `templates/grounding/system.j2`
- `OSA_CORE_PROMPT` body -> `templates/planning/system.j2`
- `SINGLE_AGENT_ROUND_PROMPT` body -> `templates/single/system.j2`

Each template must extend `shared/base_system.j2`. The concrete pattern is:

```jinja
{% extends 'shared/base_system.j2' %}
{% from 'shared/output_contracts.j2' import raw_json_contract %}
{% from 'shared/tool_discipline.j2' import no_duplicate_tool_calls %}

{% block agent_role %}
You are the <agent name> for a 5G PCF control system.
{% endblock %}

{% block responsibility %}
<agent-specific responsibility text moved from the existing Python constant>
{% endblock %}

{% block tool_discipline %}
{{ no_duplicate_tool_calls() }}
<agent-specific tool rules moved from the existing Python constant>
{% endblock %}

{% block output_contract %}
{{ raw_json_contract('<SchemaName>') }}
<agent-specific output rules moved from the existing Python constant>
{% endblock %}
```

For IEA and OSA templates, import `shared/knowledge_search.j2` and render the matching macro with the skill text passed from Python:

```jinja
{% from 'shared/knowledge_search.j2' import iea_knowledge_search %}
{% block knowledge_search %}
{{ iea_knowledge_search() }}
{% endblock %}
```

Keep the moved prompt text in templates only. Do not duplicate the full long body in Python.

- [ ] **Step 5: Render compatibility constants from Jinja**

In `grounding.py`, keep only dynamic rules and compatibility rendering:

```python
from __future__ import annotations

from .engine import PromptEngine
from .knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL


IEA_DYNAMIC_RULES = """
Dynamic grounding rules for this round:
- Treat Main's routing and retry scope from the user prompt as binding guidance.
- Preserve stable artifacts only when the evidence still supports them.
- Use cached evidence directly when it already grounds the answer.
- Call tools only when a required target is still ambiguous.
"""


def _render_system_prompt() -> str:
    return PromptEngine().render(
        "grounding/system.j2",
        iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
    )


IEA_SYSTEM_PROMPT = _render_system_prompt()
IEA_CORE_PROMPT = IEA_SYSTEM_PROMPT


__all__ = ["IEA_CORE_PROMPT", "IEA_DYNAMIC_RULES", "IEA_SYSTEM_PROMPT"]
```

Apply the same pattern to:

- `main.py`: render `main/system.j2`; set `MAIN_CONTROL_CORE_PROMPT = MAIN_CONTROL_SYSTEM_PROMPT`.
- `single.py`: render `single/system.j2`; export `SINGLE_AGENT_ROUND_PROMPT`.
- `planning.py`: render `planning/system.j2` with `OSA_KNOWLEDGE_SEARCH_SKILL`; set `OSA_CORE_PROMPT = OSA_SYSTEM_PROMPT`.

- [ ] **Step 6: Render builders through templates**

Update builder classes so they call templates, not Python constants:

```python
class GroundingPromptBuilder(PromptBuilder):
    def system_prompt(self) -> str:
        from ..knowledge_search import IEA_KNOWLEDGE_SEARCH_SKILL

        return self.render_template(
            "grounding/system.j2",
            iea_knowledge_search_skill=IEA_KNOWLEDGE_SEARCH_SKILL,
        )
```

`MainPromptBuilder.system_prompt()`:

```python
return self.render_template("main/system.j2")
```

`SinglePromptBuilder.system_prompt()`:

```python
return self.render_template("single/system.j2")
```

`PlanningPromptBuilder.system_prompt()`:

```python
from ..knowledge_search import OSA_KNOWLEDGE_SEARCH_SKILL

return self.render_template(
    "planning/system.j2",
    osa_knowledge_search_skill=OSA_KNOWLEDGE_SEARCH_SKILL,
)
```

- [ ] **Step 7: Add and use planning/user.j2**

Create `templates/planning/user.j2`:

```jinja
Structured operation intent:
{{ normalized_user_intent | json_dumps }}

Planning context:
{{ coordination_context | json_dumps }}

Planning evidence:
{{ planning_evidence | json_dumps }}

{{ tool_policy }}

{{ dynamic_rules }}

Task:
- Inspect the evidence and return one complete grounded OsaAdvisorOutput.
- If evidence is sufficient, return planning_status="executable_plan" with all required fields grounded.
- If evidence is insufficient or optimizer is infeasible/incomplete, return partial_plan or needs_upstream_reground.
- Respect control_semantics.current_stage; optimize only the active stage flows.
- Prefer optimizer sla values over telemetry values when filling final policy fields.

{{ output_format_rules }}

Return one OsaAdvisorOutput JSON object only.
```

Update `PlanningPromptBuilder.advisor_user_prompt()`:

```python
from ..planning import OSA_DYNAMIC_RULES, _OUTPUT_FORMAT_RULES, _render_round_tool_policy

return self.render_template(
    "planning/user.j2",
    normalized_user_intent=normalized_user_intent,
    coordination_context=coordination_context,
    planning_evidence=planning_evidence,
    tool_policy=_render_round_tool_policy(available_tool_names),
    dynamic_rules=OSA_DYNAMIC_RULES.strip(),
    output_format_rules=_OUTPUT_FORMAT_RULES.strip(),
)
```

Update `planning.build_advisor_user_prompt()` to delegate to the builder or to the same direct `PromptEngine().render` call. Avoid a circular import by using `PromptEngine` directly inside `planning.py`.

- [ ] **Step 8: Verify GREEN for prompt tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q tests\test_context_engineering_refactor.py
```

Expected: prompt-engine and builder tests pass. Consolidation deletion tests may still fail until Tasks 3 and 4 are done.

- [ ] **Step 9: Commit prompt rendering changes**

Run:

```powershell
git add src\control_runtime\context\prompts tests\test_context_engineering_refactor.py
git commit -m "refactor: render control prompts with jinja"
```

---

### Task 3: Consolidate Grounding Helper Modules

**Files:**
- Modify: `src/control_runtime/agents/grounding/common.py`
- Modify: `src/control_runtime/agents/grounding/compiler.py`
- Modify: `src/control_runtime/agents/grounding/artifact_compiler.py`
- Delete: `src/control_runtime/agents/grounding/directives.py`
- Delete: `src/control_runtime/agents/grounding/qos_envelope_builder.py`
- Modify local ignored test imports if needed for the local full suite.

- [ ] **Step 1: Move MainDirectiveExtractor into common.py**

Append this import to `common.py`:

```python
import json
```

If `json` is already present after sorting imports, do not duplicate it.

Append the complete `MainDirectiveExtractor` class from `directives.py` below `normalize_domain_evidence()`. Remove the now-redundant import in the moved class:

```python
from .common import normalize_domain_evidence, normalize_requested_domains
```

The class should call the local functions directly.

- [ ] **Step 2: Move QosEnvelopeBuilder into common.py**

Add `QosTargetEnvelope` to the domain import in `common.py`:

```python
from ...domain.policy_plan import FlowSelector, QosTargetEnvelope
```

If `FlowSelector` is not currently imported in `common.py`, add both names. Append the complete `QosEnvelopeBuilder` class from `qos_envelope_builder.py` near other helper classes.

- [ ] **Step 3: Update imports**

In `grounding/compiler.py`, replace:

```python
from .directives import MainDirectiveExtractor
```

with:

```python
from .common import AM_GROUNDING_TOOLS, SM_GROUNDING_TOOLS, VALID_DOMAINS, MainDirectiveExtractor, uses_am_grounding, uses_sm_grounding
```

Then remove `MainDirectiveExtractor` from any separate import line.

In `grounding/artifact_compiler.py`, replace:

```python
from .qos_envelope_builder import QosEnvelopeBuilder
```

with an import from `.common`:

```python
from .common import (
    QosEnvelopeBuilder,
    classify_domain_resolution,
    flow_id_is_grounded,
    flow_name_is_grounded,
    normalize_domain_evidence,
    normalize_requested_domains,
    uses_am_grounding,
    uses_sm_grounding,
)
```

- [ ] **Step 4: Delete old helper files**

Use `apply_patch` delete hunks for:

```text
src/control_runtime/agents/grounding/directives.py
src/control_runtime/agents/grounding/qos_envelope_builder.py
```

- [ ] **Step 5: Update local ignored test imports if full pytest requires it**

If `pytest -q` fails because `tests/test_iea_osa_contract_regressions.py` imports `control_runtime.agents.grounding.directives`, change that local test import to:

```python
from control_runtime.agents.grounding.common import MainDirectiveExtractor
```

Do not force-add ignored tests unless the user explicitly asks to track them. The committed regression test in `tests/test_context_engineering_refactor.py` is the durable guard.

- [ ] **Step 6: Verify grounding consolidation**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q tests\test_context_engineering_refactor.py
rg -n "agents\.grounding\.directives|agents\.grounding\.qos_envelope_builder|from \.directives|from \.qos_envelope_builder" src tests -g *.py
```

Expected: context test no longer fails on the two deleted grounding files. `rg` should produce no matches except intentionally updated ignored tests if they still contain local comments; remove those comments if present.

- [ ] **Step 7: Commit grounding consolidation**

Run:

```powershell
git add src\control_runtime\agents\grounding tests\test_context_engineering_refactor.py
git commit -m "refactor: consolidate grounding helpers"
```

---

### Task 4: Consolidate Planning Policy Normalizer

**Files:**
- Modify: `src/control_runtime/agents/planning/planning_artifact_compiler.py`
- Modify: `src/control_runtime/agents/planning/planning_validation.py`
- Modify: `src/control_runtime/agents/planning/tools.py`
- Modify: `src/control_runtime/agents/planning/agent.py`
- Modify: `src/control_runtime/agents/single/agent.py`
- Delete: `src/control_runtime/agents/planning/policy_normalizer.py`

- [ ] **Step 1: Move normalizer imports into planning_artifact_compiler.py**

At the top of `planning_artifact_compiler.py`, add imports currently used by `policy_normalizer.py`:

```python
from datetime import date, datetime
from enum import Enum
import re

from pydantic import BaseModel

from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest
```

- [ ] **Step 2: Move normalizer functions into planning_artifact_compiler.py**

Move the complete current definitions listed below from `policy_normalizer.py` to `planning_artifact_compiler.py`, placing them above `class PlanningArtifactCompiler`. Preserve each function body exactly during the move; this task is relocation only.

```python
def json_friendly(value: Any) -> Any
def normalize_app_id(app_id: Any) -> str
def _require_policy_id(policy_id: Any, *, policy_type: str) -> str
def _require_supi(supi: Any) -> str
def _normalize_sm_policy_details(details: Dict[str, Any], *, flow_id: str, app_id: str) -> Dict[str, Any]
def _normalize_ursp_policy_details(details: Dict[str, Any], *, target_type: str, flow_id: str | None) -> Dict[str, Any]
def _normalize_am_policy_details(details: Dict[str, Any], *, supi: str) -> Dict[str, Any]
def normalize_policy_plan_draft(draft: PolicyPlanDraft) -> PolicyPlanDraft
```

After moving, `policy_normalizer.py` should contain no remaining function definitions and will be deleted in Step 5.

- [ ] **Step 3: Replace private aliases in planning_artifact_compiler.py**

Delete these imports:

```python
from .policy_normalizer import json_friendly as _json_friendly
from .policy_normalizer import normalize_app_id as _normalize_app_id
from .policy_normalizer import normalize_policy_plan_draft
```

Add local aliases below `normalize_policy_plan_draft()` to minimize diff size:

```python
_json_friendly = json_friendly
_normalize_app_id = normalize_app_id
```

Keep the rest of `PlanningArtifactCompiler` unchanged for now.

- [ ] **Step 4: Update dependent imports**

In `planning_validation.py`, replace:

```python
from .policy_normalizer import normalize_app_id as _normalize_app_id
```

with:

```python
from .planning_artifact_compiler import normalize_app_id as _normalize_app_id
```

In `planning/tools.py`, replace:

```python
from .policy_normalizer import json_friendly as _json_friendly
```

with:

```python
from .planning_artifact_compiler import json_friendly as _json_friendly
```

In `planning/agent.py`, replace:

```python
from .policy_normalizer import normalize_app_id as _normalize_app_id
```

with:

```python
from .planning_artifact_compiler import normalize_app_id as _normalize_app_id
```

In `single/agent.py`, replace:

```python
from ..planning.policy_normalizer import normalize_policy_plan_draft
```

with:

```python
from ..planning.planning_artifact_compiler import normalize_policy_plan_draft
```

- [ ] **Step 5: Delete policy_normalizer.py**

Use an `apply_patch` delete hunk for:

```text
src/control_runtime/agents/planning/policy_normalizer.py
```

- [ ] **Step 6: Break circular imports if they appear**

Run the context test. If Python reports a circular import between `planning_artifact_compiler.py` and `planning_validation.py`, split the moved helper functions into the top of `planning_validation.py` instead and import them from there. Prefer keeping them in `planning_artifact_compiler.py` because the normalization is artifact-facing, but do not keep a separate `policy_normalizer.py` just to avoid a small import edit.

- [ ] **Step 7: Verify planning consolidation**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q tests\test_context_engineering_refactor.py
rg -n "agents\.planning\.policy_normalizer|from \.policy_normalizer|planning\.policy_normalizer" src tests -g *.py
```

Expected: context test passes, and `rg` produces no matches.

- [ ] **Step 8: Commit planning consolidation**

Run:

```powershell
git add src\control_runtime\agents\planning src\control_runtime\agents\single\agent.py tests\test_context_engineering_refactor.py
git commit -m "refactor: consolidate planning policy normalization"
```

---

### Task 5: Final Verification and Code-Volume Check

**Files:**
- No planned code edits unless verification reveals a failure.

- [ ] **Step 1: Run focused prompt and contract tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q tests\test_context_engineering_refactor.py
```

Expected: all tests in that file pass.

- [ ] **Step 2: Run full local test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: full local suite passes. If ignored local tests fail because they import deleted helper modules, update their imports locally to the new modules and rerun.

- [ ] **Step 3: Run deleted-module scan**

Run:

```powershell
rg -n "agents\.grounding\.directives|agents\.grounding\.qos_envelope_builder|agents\.planning\.policy_normalizer|from \.directives|from \.qos_envelope_builder|from \.policy_normalizer" src tests -g *.py
```

Expected: no matches.

- [ ] **Step 4: Measure Python code reduction in target areas**

Run before final response:

```powershell
@(
  'src/control_runtime/context/prompts',
  'src/control_runtime/agents/grounding',
  'src/control_runtime/agents/planning'
) | ForEach-Object {
  Get-ChildItem $_ -Recurse -File -Filter *.py
} | ForEach-Object {
  (Get-Content $_.FullName | Measure-Object -Line).Lines
} | Measure-Object -Sum
```

Also report deleted Python files:

```powershell
git show --stat --oneline HEAD~3..HEAD -- src/control_runtime/context/prompts src/control_runtime/agents/grounding src/control_runtime/agents/planning
```

Expected: target Python file count drops by at least 3, and Python prompt facade files are shorter. If total repository text grows due templates, explain that Python code volume and module count were reduced while prompt text moved to `.j2`.

- [ ] **Step 5: Inspect git status**

Run:

```powershell
git status --short --branch
```

Expected: only pre-existing unrelated dirty files remain. If implementation files are dirty, stage and commit them or fix the issue before final response.

- [ ] **Step 6: Final commit if verification fixes were required**

If verification required small fixes, commit them:

```powershell
git add src\control_runtime\context\prompts src\control_runtime\agents\grounding src\control_runtime\agents\planning src\control_runtime\agents\single\agent.py tests\test_context_engineering_refactor.py
git commit -m "test: verify prompt consolidation contracts"
```

Do not include unrelated dirty files.

---

## Self-Review Notes

- Spec coverage: Jinja strict rendering, shared templates, builder rendering, compatibility constants, grounding helper consolidation, planning normalizer consolidation, and verification are each covered by a task.
- Red-flag scan: The plan contains no unresolved marker text. The planning normalizer task names exact functions to move and exact source/target files.
- Type consistency: The plan keeps current function names `json_friendly`, `normalize_app_id`, and `normalize_policy_plan_draft`; callers are updated to import those names from `planning_artifact_compiler.py`.
- Scope: The plan intentionally avoids merging large compiler/validator/tool files and avoids unrelated environment or experiment work.
