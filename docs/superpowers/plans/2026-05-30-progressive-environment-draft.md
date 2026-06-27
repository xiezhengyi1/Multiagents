# Progressive Environment Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-scenario environment generation with an in-memory progressive draft workflow that supports ordered section replacement, focused entity patches, on-demand inspection, gated YAML writing, and gated simulation.

**Architecture:** Add a focused `ScenarioDraftStore` that owns one in-memory draft and exposes deterministic state transitions. Keep final static schema validation in `EnvironmentAgentCompiler`; adapt `build_environment_tools()` into a thin tool layer over the store. Update the environment prompt so the LLM generates bounded sections and requests full section context only when needed.

**Tech Stack:** Python 3.12, dataclasses, pathlib, existing LangChain structured tools, unittest/pytest.

---

## File Structure

- Create `src/environment/draft.py`: in-memory draft state, stage ordering, assembly, summaries, replacement, and stable-ID patching.
- Modify `src/environment/tools.py`: expose progressive draft tools and gate simulation behind a validated written draft.
- Modify `src/environment/prompts.py`: require the progressive workflow and prohibit full-scenario repair payloads.
- Modify `src/environment/agent.py`: reload the written YAML when constructing
  the returned candidate so final model output remains compact.
- Modify `agent_runtime/execution/structured_tool_loop.py`: allow selected tools
  to override the default per-tool call limit.
- Modify `src/shared/agents/base.py`: forward selected tool call limits.
- Modify `tests/test_environment_agent.py`: add store, tool, prompt, and gating regression tests.

### Task 1: Add Ordered In-Memory Draft State

**Files:**
- Create: `src/environment/draft.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Write failing tests for initialization, ordered first writes, and upstream replacement**

Add:

```python
from environment.draft import ScenarioDraftStore


class ScenarioDraftStoreTest(unittest.TestCase):
    def test_replace_section_requires_initialization(self) -> None:
        store = ScenarioDraftStore()

        with self.assertRaisesRegex(ValueError, "draft is not initialized"):
            store.replace_section("slices", [])

    def test_first_section_write_must_follow_declared_order(self) -> None:
        store = ScenarioDraftStore()
        store.initialize({"name": "G001", "scenario_id": "G001", "tick_ms": 100, "seed": 7})

        with self.assertRaisesRegex(ValueError, "next required section is slices"):
            store.replace_section("upfs", [{"name": "upf"}])

    def test_completed_upstream_section_can_be_replaced(self) -> None:
        store = ScenarioDraftStore()
        store.initialize({"name": "G001", "scenario_id": "G001", "tick_ms": 100, "seed": 7})
        store.replace_section("slices", [{"label": "slice-2-000001"}])
        store.replace_section("upfs", [{"name": "upf"}])

        summary = store.replace_section("slices", [{"label": "slice-1-000001"}])

        self.assertEqual(summary["completed_sections"], ["metadata", "slices", "upfs"])
        self.assertEqual(summary["next_section"], "gnbs")
        self.assertEqual(summary["reference_ids"]["slice_labels"], ["slice-1-000001"])
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_environment_agent.py::ScenarioDraftStoreTest -q
```

Expected: FAIL because `environment.draft` does not exist.

- [ ] **Step 3: Implement minimal draft store**

Create `src/environment/draft.py`:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .contracts import ScenarioCandidate


SECTION_ORDER = (
    "metadata",
    "slices",
    "upfs",
    "gnbs",
    "ues",
    "apps",
    "flows",
    "runtime_config",
    "split_mode_overlay",
)
COLLECTION_ID_KEYS = {
    "slices": "label",
    "upfs": "name",
    "gnbs": "name",
    "ues": "name",
    "apps": "app_id",
    "flows": "flow_id",
}


@dataclass
class ScenarioDraftStore:
    sections: dict[str, Any] = field(default_factory=dict)
    completed_sections: list[str] = field(default_factory=list)
    validation_passed: bool = False
    scenario_path: str = ""
    split_mode_overlay_path: str = ""

    def initialize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        required = ("name", "scenario_id", "tick_ms", "seed")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError("metadata missing fields: " + ", ".join(missing))
        self.sections = {"metadata": deepcopy(metadata)}
        self.completed_sections = ["metadata"]
        self._invalidate_evidence()
        return self.summary()

    def replace_section(self, section: str, payload: Any) -> dict[str, Any]:
        self._require_initialized()
        self._require_known_section(section)
        if section == "metadata":
            raise ValueError("metadata must be set with initialize")
        next_section = self.next_section()
        if section not in self.completed_sections and section != next_section:
            raise ValueError(f"next required section is {next_section}")
        self.sections[section] = deepcopy(payload)
        if section not in self.completed_sections:
            self.completed_sections.append(section)
        self._invalidate_evidence()
        return self.summary()

    def next_section(self) -> str | None:
        for section in SECTION_ORDER:
            if section not in self.completed_sections:
                return section
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": str((self.sections.get("metadata") or {}).get("scenario_id") or ""),
            "completed_sections": list(self.completed_sections),
            "next_section": self.next_section(),
            "section_counts": {
                key: len(value)
                for key, value in self.sections.items()
                if isinstance(value, list)
            },
            "reference_ids": self._reference_ids(),
            "validation_passed": self.validation_passed,
            "scenario_path": self.scenario_path,
        }

    def _reference_ids(self) -> dict[str, list[str]]:
        labels = {
            "slices": "slice_labels",
            "upfs": "upf_names",
            "gnbs": "gnb_names",
            "ues": "ue_names",
            "apps": "app_ids",
            "flows": "flow_ids",
        }
        return {
            labels[section]: [
                str(item.get(id_key) or "")
                for item in self.sections.get(section, [])
                if isinstance(item, dict) and str(item.get(id_key) or "")
            ]
            for section, id_key in COLLECTION_ID_KEYS.items()
            if section in self.sections
        }

    def _invalidate_evidence(self) -> None:
        self.validation_passed = False
        self.scenario_path = ""
        self.split_mode_overlay_path = ""

    def _require_initialized(self) -> None:
        if "metadata" not in self.sections:
            raise ValueError("draft is not initialized")

    @staticmethod
    def _require_known_section(section: str) -> None:
        if section not in SECTION_ORDER:
            raise ValueError(f"unsupported draft section: {section}")
```

- [ ] **Step 4: Run store tests to verify green**

Run the command from Step 2.

Expected: PASS.

### Task 2: Add Stable-ID Patch, Inspection, and Assembly

**Files:**
- Modify: `src/environment/draft.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
    def test_patch_updates_existing_flow_without_adding_duplicate(self) -> None:
        store = _complete_draft_store()

        summary = store.patch_entity("flows", "flow-1", {"name": "updated"})

        self.assertEqual(summary["section_counts"]["flows"], 1)
        self.assertEqual(store.inspect_section("flows")["payload"][0]["name"], "updated")

    def test_patch_rejects_unknown_entity(self) -> None:
        store = _complete_draft_store()

        with self.assertRaisesRegex(ValueError, "flow missing-flow does not exist"):
            store.patch_entity("flows", "missing-flow", {"name": "updated"})

    def test_inspect_discloses_only_requested_section(self) -> None:
        store = _complete_draft_store()

        inspected = store.inspect_section("flows")

        self.assertIn("payload", inspected)
        self.assertEqual(inspected["section"], "flows")
        self.assertNotIn("slices", inspected["payload"])

    def test_assemble_candidate_uses_runtime_config_and_overlay(self) -> None:
        store = _complete_draft_store()

        candidate = store.assemble_candidate()

        self.assertEqual(candidate.scenario["flows"][0]["flow_id"], "flow-1")
        self.assertEqual(candidate.scenario["free5gc"]["mode"], "single_upf")
```

Add a helper that initializes all sections using `_minimal_scenario()` and
`split_mode_overlay=None`.

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_environment_agent.py::ScenarioDraftStoreTest -q
```

Expected: FAIL because patch, inspect, and assembly methods do not exist.

- [ ] **Step 3: Implement patch, inspect, and assembly**

Add to `ScenarioDraftStore`:

```python
    def patch_entity(self, section: str, entity_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        id_key = COLLECTION_ID_KEYS.get(section)
        if not id_key:
            raise ValueError(f"section {section} does not support entity patches")
        entities = self.sections.get(section)
        if not isinstance(entities, list):
            raise ValueError(f"section {section} has not been generated")
        for entity in entities:
            if isinstance(entity, dict) and str(entity.get(id_key) or "") == str(entity_id):
                entity.update(deepcopy(changes))
                self._invalidate_evidence()
                return self.summary()
        entity_name = id_key.removesuffix("_id")
        raise ValueError(f"{entity_name} {entity_id} does not exist")

    def inspect_section(self, section: str) -> dict[str, Any]:
        self._require_initialized()
        self._require_known_section(section)
        return {
            "status": "ok",
            "section": section,
            "payload": deepcopy(self.sections.get(section)),
            "draft": self.summary(),
        }

    def assemble_candidate(self) -> ScenarioCandidate:
        missing = [section for section in SECTION_ORDER if section not in self.completed_sections]
        if missing:
            raise ValueError("draft missing sections: " + ", ".join(missing))
        metadata = deepcopy(self.sections["metadata"])
        runtime = deepcopy(self.sections["runtime_config"])
        scenario = {
            **metadata,
            "slices": deepcopy(self.sections["slices"]),
            "upfs": deepcopy(self.sections["upfs"]),
            "gnbs": deepcopy(self.sections["gnbs"]),
            "ues": deepcopy(self.sections["ues"]),
            "apps": deepcopy(self.sections["apps"]),
            "flows": deepcopy(self.sections["flows"]),
            **runtime,
        }
        return ScenarioCandidate(
            scenario_id=str(metadata["scenario_id"]),
            name=str(metadata["name"]),
            scenario=scenario,
            split_mode_overlay=deepcopy(self.sections["split_mode_overlay"]),
        )
```

- [ ] **Step 4: Run store tests**

Expected: PASS.

### Task 3: Expose Progressive Draft Tools and Validation Gate

**Files:**
- Modify: `src/environment/tools.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Write failing tool-surface and validation-gate tests**

Replace the old expected tool names and add tests:

```python
    def test_environment_agent_exposes_progressive_draft_tools(self) -> None:
        names = {tool.name for tool in _build_test_environment_tools()}

        self.assertEqual(
            names,
            {
                "list_existing_environment_specs",
                "initialize_environment_draft",
                "replace_draft_section",
                "patch_draft_entity",
                "inspect_draft_section",
                "validate_environment_draft",
                "write_validated_environment_yaml",
                "simulate_candidate_environment",
                "record_validation_feedback",
            },
        )

    def test_write_rejects_draft_before_validation_passes(self) -> None:
        tools = _build_test_environment_tools()
        _populate_tools_with_minimal_draft(tools)
        tool = next(item for item in tools if item.name == "write_validated_environment_yaml")

        with self.assertRaisesRegex(ValueError, "draft must pass validation before writing YAML"):
            tool.invoke({})
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_environment_agent.py::EnvironmentAgentToolingTest -q
```

Expected: FAIL because progressive tools are not exposed.

- [ ] **Step 3: Add `ScenarioDraftStore` to tool construction**

In `src/environment/tools.py`, import and instantiate the store:

```python
from .draft import ScenarioDraftStore

draft = ScenarioDraftStore()
```

Add thin tools:

```python
    def initialize_environment_draft(reason: str = "", metadata: dict[str, Any] | None = None) -> str:
        return _json({"status": "ok", "reason": reason, "draft": draft.initialize(metadata or {})})

    def replace_draft_section(reason: str = "", section: str = "", payload: Any = None) -> str:
        return _json({"status": "ok", "reason": reason, "draft": draft.replace_section(section, payload)})

    def patch_draft_entity(
        reason: str = "",
        section: str = "",
        entity_id: str = "",
        changes: dict[str, Any] | None = None,
    ) -> str:
        return _json({"status": "ok", "reason": reason, "draft": draft.patch_entity(section, entity_id, changes or {})})

    def inspect_draft_section(reason: str = "", section: str = "") -> str:
        return _json({"reason": reason, **draft.inspect_section(section)})

    def validate_environment_draft(reason: str = "") -> str:
        candidate = draft.assemble_candidate()
        report = compiler.validate_candidate(candidate)
        draft.validation_passed = report.ok
        return _json({
            "status": "ok" if report.ok else "failed",
            "reason": reason,
            "scenario_id": report.scenario_id,
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "draft": draft.summary(),
        })
```

- [ ] **Step 4: Run tooling tests**

Expected: tool-surface tests pass; write gate remains red until Task 4.

### Task 4: Gate YAML Writing and Simulation

**Files:**
- Modify: `src/environment/draft.py`
- Modify: `src/environment/tools.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Write failing tests for valid write and simulation gate**

Add:

```python
    def test_validated_draft_writes_absolute_yaml_path(self) -> None:
        tools = _build_test_environment_tools()
        _populate_tools_with_minimal_draft(tools)
        _invoke_tool(tools, "validate_environment_draft", {})

        result = _invoke_tool(tools, "write_validated_environment_yaml", {})

        self.assertTrue(Path(result["scenario_path"]).is_absolute())

    def test_simulation_rejects_unwritten_draft(self) -> None:
        tools = _build_test_environment_tools()
        _populate_tools_with_minimal_draft(tools)
        _invoke_tool(tools, "validate_environment_draft", {})
        simulate = next(item for item in tools if item.name == "simulate_candidate_environment")

        with self.assertRaisesRegex(ValueError, "validated YAML must be written before simulation"):
            simulate.invoke({})
```

- [ ] **Step 2: Run tests to verify red**

Expected: FAIL because write and simulation gates are not implemented.

- [ ] **Step 3: Implement write evidence recording**

Add to `ScenarioDraftStore`:

```python
    def record_written_paths(self, *, scenario_path: Path, split_mode_overlay_path: Path | None) -> None:
        self.scenario_path = str(Path(scenario_path).resolve())
        self.split_mode_overlay_path = (
            str(Path(split_mode_overlay_path).resolve()) if split_mode_overlay_path else ""
        )
```

Replace complete scenario write tool with:

```python
    def write_validated_environment_yaml(reason: str = "", output_dir: str = "") -> str:
        if not draft.validation_passed:
            raise ValueError("draft must pass validation before writing YAML")
        candidate = draft.assemble_candidate()
        filename = candidate.scenario_id.lower().replace("-", "_") + ".yaml"
        base_path = _resolve_candidate_output_dir(Path(scenario_root), output_dir) / filename
        _dump_mapping(base_path, candidate.scenario)
        overlay_path = None
        if candidate.split_mode_overlay is not None:
            overlay_path = Path(scenario_root) / "split_mode" / filename
            _dump_mapping(overlay_path, candidate.split_mode_overlay)
        draft.record_written_paths(scenario_path=base_path, split_mode_overlay_path=overlay_path)
        return _json({
            "status": "ok",
            "reason": reason,
            "scenario_id": candidate.scenario_id,
            "scenario_path": draft.scenario_path,
            "split_mode_overlay_path": draft.split_mode_overlay_path,
        })
```

Change simulation to use draft write evidence and reject caller-supplied paths:

```python
        if not draft.scenario_path:
            raise ValueError("validated YAML must be written before simulation")
        plan = launcher.build_direct_launch_plan(
            scenario_path=Path(draft.scenario_path),
```

- [ ] **Step 4: Run tooling tests**

Expected: PASS.

### Task 5: Update Prompt Contract

**Files:**
- Modify: `src/environment/prompts.py`
- Modify: `src/environment/agent.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Write failing prompt test**

Add assertions:

```python
        self.assertIn("initialize_environment_draft", prompt)
        self.assertIn("replace_draft_section", prompt)
        self.assertIn("patch_draft_entity", prompt)
        self.assertIn("inspect_draft_section", prompt)
        self.assertIn("Do not submit a complete scenario mapping", prompt)
```

Update `test_agent_builds_tool_driven_loop_with_environment_tools` with the same
tool-name expectations against `agent.system_prompt`.

- [ ] **Step 2: Run prompt tests to verify red**

Expected: FAIL because the old prompt still requires full scenario generation.

- [ ] **Step 3: Rewrite prompt workflow**

Replace the mandatory loop with progressive tool instructions:

```text
1. Call list_existing_environment_specs.
2. Call initialize_environment_draft with metadata only.
3. Populate slices, upfs, gnbs, ues, apps, flows, runtime_config, and
   split_mode_overlay in order using replace_draft_section.
4. Use patch_draft_entity for focused repairs to completed collection sections.
5. Use inspect_draft_section only when a compact summary is insufficient.
6. Call validate_environment_draft.
7. After validation status is ok, call write_validated_environment_yaml.
8. After YAML write succeeds, call simulate_candidate_environment.
9. Record feedback and repair the affected section when validation or simulation
   fails.
```

Add:

```text
Do not submit a complete scenario mapping during initialization, replacement, or
patch calls. Generate and repair only the bounded section needed for the current
stage.
```

Update `src/environment/agent.py` closed-loop requirement to name the new tools.

- [ ] **Step 4: Run prompt tests**

Expected: PASS.

### Task 6: Keep Final Advisor Output Compact

**Files:**
- Modify: `src/environment/agent.py`
- Test: `tests/test_environment_agent.py`

- [ ] **Step 1: Add tests for compact output and written YAML reload**

Verify `EnvironmentAdvisorOutput` validates without a scenario mapping and that
`_load_written_candidate()` reads the YAML path emitted by the write tool.

- [ ] **Step 2: Remove repeated scenario fields from advisor output**

Keep only `scenario_id`, `name`, validation status, feedback, summary, and
rationale in the model response.

- [ ] **Step 3: Construct the returned candidate from write evidence**

Extract the last `write_validated_environment_yaml` payload, reload its absolute
scenario path and optional overlay path, and validate the reconstructed
candidate before returning it.

- [ ] **Step 4: Run focused environment tests**

Expected: PASS.

### Task 7: Verify Progressive Environment Generation

**Files:**
- Test: `tests/test_environment_agent.py`
- Test: `tests/test_structured_tool_loop_parser.py`

- [ ] **Step 1: Run focused regression tests**

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_environment_agent.py tests/test_structured_tool_loop_parser.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/environment tests/test_environment_agent.py
```

Expected: exit code `0`.

- [ ] **Step 3: Check patch hygiene**

```powershell
git diff --check -- src/environment tests/test_environment_agent.py
```

Expected: exit code `0`.

- [ ] **Step 4: Run the Linux integration check on `nccl3`**

```bash
python src/environment.py
```

Expected: trace shows progressive section tools, static validation succeeds
before YAML writing, and simulation receives the written absolute scenario path.
