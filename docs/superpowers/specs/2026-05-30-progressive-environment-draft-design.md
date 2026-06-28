# Progressive Environment Draft Design

## Problem

The environment generation agent currently asks the model to submit a complete
scenario mapping on every repair attempt. A medium scenario contains slices,
UPFs, gNBs, UEs, applications, flows, runtime configuration, and an optional
split-mode overlay. Re-sending that full mapping consumes context, encourages
duplicate filler, and makes local repairs drift into unrelated fields.

The agent needs progressive disclosure: generate one bounded section at a time,
return compact summaries by default, expose full section content only on
request, and assemble the final YAML only after static validation passes.

## Decisions

- Keep one in-memory draft per `build_environment_tools()` invocation.
- Do not persist incomplete drafts to disk.
- Advance through an ordered workflow while allowing edits to completed
  sections.
- Support both whole-section replacement and stable-ID entity patching.
- Return compact summaries after mutation.
- Add an explicit inspection tool for full disclosure of one section.
- Preserve the existing compiler as the final static validator.
- Write YAML only after successful draft validation.
- Simulate only after YAML has been written.
- Keep the model's final JSON compact; reload the written YAML in the runtime
  when constructing the returned `ScenarioCandidate`.

## Draft Model

Add `src/environment/draft.py` with `ScenarioDraftStore`.

The store owns:

- `scenario_id`
- `name`
- ordered section payloads
- completed section names
- validation status
- written scenario path

Ordered sections:

```text
metadata
slices
upfs
gnbs
ues
apps
flows
runtime_config
split_mode_overlay
```

`metadata` contains scalar scenario fields:

```text
name, scenario_id, tick_ms, seed
```

`runtime_config` contains:

```text
free5gc, ns3, writer, topology, bridge
```

`split_mode_overlay` is optional but remains an explicit stage. It may be set to
`null` when no overlay is needed.

Collection sections use stable entity identifiers:

```text
slices -> label
upfs   -> name
gnbs   -> name
ues    -> name
apps   -> app_id
flows  -> flow_id
```

## State Rules

`initialize_environment_draft` creates a fresh draft and records metadata.

The first write to a section must follow the declared order. A completed section
may be replaced or patched later. Revising an upstream section invalidates the
previous validation result and written path, but it does not erase downstream
content automatically.

`patch_draft_entity` updates an existing collection entity by stable ID. It does
not insert or delete entities. Structural additions and deletions use
`replace_draft_section`, making changes explicit.

`validate_environment_draft` assembles a `ScenarioCandidate` and delegates to
`EnvironmentAgentCompiler.validate_candidate()`. Validation fails early when a
required stage has not been completed.

`write_validated_environment_yaml` refuses to write unless the latest draft
validation passed. The write operation emits an absolute scenario path and
optional absolute split-mode overlay path.

`simulate_candidate_environment` accepts only the scenario path already emitted
by the write tool. Existing simulator readiness checks remain unchanged.

After successful simulation, the model returns only `scenario_id`, `name`,
`validation_status`, `validation_feedback`, `tool_loop_summary`, and
`rationale`. `EnvironmentGenerationAgent` reloads the YAML path emitted by
`write_validated_environment_yaml`; the model never repeats the complete
scenario in its final response.

## Tool Surface

Retain:

```text
list_existing_environment_specs
simulate_candidate_environment
record_validation_feedback
```

Replace the complete-scenario write and validate workflow with:

```text
initialize_environment_draft
replace_draft_section
patch_draft_entity
inspect_draft_section
validate_environment_draft
write_validated_environment_yaml
```

Mutation tools return:

```json
{
  "status": "ok",
  "draft": {
    "scenario_id": "G001",
    "completed_sections": ["metadata", "slices"],
    "next_section": "upfs",
    "section_counts": {"slices": 3},
    "reference_ids": {"slice_labels": ["slice-1-000001"]}
  }
}
```

They do not return complete entity bodies.

`inspect_draft_section(section)` returns only the requested section and the
compact draft summary.

## Prompt Workflow

Update the environment agent prompt to require this sequence:

```text
list specs
initialize draft
replace slices
replace upfs
replace gnbs
replace ues
replace apps
replace flows
replace runtime_config
replace split_mode_overlay
validate draft
write validated YAML
simulate
return final JSON
```

The prompt must state:

- Use section replacement for initial generation.
- Use entity patching for focused repairs.
- Inspect only the section required for diagnosis.
- Do not submit a full scenario mapping during mutation calls.
- Do not write YAML before validation succeeds.
- Do not simulate before the write tool emits the scenario path.

## Error Handling

Tools return precise errors for:

- missing draft initialization
- skipped initial stage
- unsupported section name
- patching a non-collection section
- missing stable entity ID
- patching an entity that does not exist
- validation before all required stages are complete
- writing before validation succeeds
- simulation before a written scenario exists

After any mutation, prior validation and write evidence are invalidated.

## Testing

Extend `tests/test_environment_agent.py` with focused store and tool tests:

- ordered first-write enforcement
- replacement of a completed upstream section
- patching one flow without adding duplicates
- inspection disclosure limited to one requested section
- validation rejection for incomplete drafts
- successful assembly accepted by the existing compiler
- write rejection before successful validation
- write returns absolute YAML path
- simulation rejection before draft write
- tool surface exposes the progressive workflow
- prompt requires progressive generation and forbids full-scenario repair

Existing launcher, SLA initialization, spec explorer, and final simulation
evidence tests remain in place.
