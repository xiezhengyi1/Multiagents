from __future__ import annotations

from pathlib import Path

import pytest

from shared.contracts.coordination import (
    HandoffPayload,
    ProposalPayload,
    RepairRequestPayload,
    TerminalPayload,
    ValidationIssue,
    ValidationResultPayload,
)
from shared.contracts.model_trace import ModelMessage
from shared.runtime.journal import EpisodeJournal
from training import EpisodeRecords, project_rl_trajectory, project_sft_examples


class SampleDecision(ProposalPayload):
    value: str


class SampleHandoff(HandoffPayload):
    label: str


def _append_success_trace(
    *,
    journal: EpisodeJournal,
    trigger_event_id: str,
    agent_name: str,
    value: str,
) -> None:
    journal.append_trace(
        trigger_event_id=trigger_event_id,
        round_index=0,
        agent_name=agent_name,
        model_name="test-model",
        input_messages=[ModelMessage(role="user", content="decide")],
        output_message=ModelMessage(role="assistant", content={"value": value}),
        tools=[],
        status="success",
    )


def test_native_episode_records_select_only_validator_confirmed_sft_calls(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-training")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="main",
        message_type="handoff",
        payload=SampleHandoff(label="request"),
    )
    _append_success_trace(
        journal=journal,
        trigger_event_id=trigger.event_id,
        agent_name="main",
        value="bad",
    )
    invalid = journal.append_event(
        round_index=0,
        source_agent="main",
        target_agent="sba",
        message_type="proposal",
        in_reply_to=trigger.event_id,
        input_artifact_ids=[trigger.event_id],
        payload=SampleDecision(value="bad"),
    )
    repair = journal.append_event(
        round_index=0,
        source_agent="main_validator",
        target_agent="main",
        message_type="repair_request",
        in_reply_to=invalid.event_id,
        invalidated_artifact_ids=[invalid.event_id],
        payload=RepairRequestPayload(
            issues=[ValidationIssue(code="bad", message="repair it")],
            attempt=1,
            max_attempts=2,
        ),
    )
    _append_success_trace(
        journal=journal,
        trigger_event_id=repair.event_id,
        agent_name="main",
        value="good",
    )
    repaired = journal.append_event(
        round_index=0,
        source_agent="main",
        target_agent="sba",
        message_type="repair_response",
        in_reply_to=repair.event_id,
        input_artifact_ids=[repair.event_id],
        payload=SampleDecision(value="good"),
    )
    accepted = journal.append_event(
        round_index=0,
        source_agent="main_validator",
        target_agent="sba",
        message_type="validation_result",
        in_reply_to=repaired.event_id,
        input_artifact_ids=[repaired.event_id],
        payload=ValidationResultPayload(notes=["accepted"]),
    )
    journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="episode_controller",
        message_type="terminal",
        in_reply_to=accepted.event_id,
        input_artifact_ids=[accepted.event_id],
        payload=TerminalPayload(
            outcome="success",
            reason="done",
            reward_total=1.0,
            reward_components={"objective": 1.0},
        ),
    )

    records = EpisodeRecords.load(journal.directory)
    candidates = records.sft_candidates()

    assert len(records.traces) == 2
    assert len(candidates) == 1
    assert candidates[0].output_event_id == repaired.event_id
    assert candidates[0].trace.output_message.content == {"value": "good"}
    assert candidates[0].disposition == "accepted"

    sft_examples = project_sft_examples(records)
    rl_trajectory = project_rl_trajectory(records)
    assert len(sft_examples) == 1
    assert sft_examples[0].example_id == candidates[0].trace.trace_id
    assert sft_examples[0].output_message.content == {"value": "good"}
    assert rl_trajectory.episode_id == "episode-training"
    assert rl_trajectory.events == records.events
    assert rl_trajectory.traces == records.traces
    assert rl_trajectory.terminal.outcome == "success"


def test_valid_upstream_repair_is_sft_eligible_without_invalidating_model_output(
    tmp_path: Path,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode-upstream-training")
    trigger = journal.append_event(
        round_index=0,
        source_agent="sba_compiler",
        target_agent="ppa",
        message_type="handoff",
        payload=SampleHandoff(label="grounding"),
    )
    _append_success_trace(
        journal=journal,
        trigger_event_id=trigger.event_id,
        agent_name="ppa",
        value="need-binding",
    )
    proposal = journal.append_event(
        round_index=0,
        source_agent="ppa",
        target_agent="sba",
        message_type="proposal",
        in_reply_to=trigger.event_id,
        input_artifact_ids=[trigger.event_id],
        payload=SampleDecision(value="need-binding"),
    )
    repair = journal.append_event(
        round_index=0,
        source_agent="ppa_validator",
        target_agent="sba",
        message_type="repair_request",
        in_reply_to=proposal.event_id,
        input_artifact_ids=[proposal.event_id],
        payload=RepairRequestPayload(
            issues=[ValidationIssue(code="binding", message="rebind upstream")],
            attempt=1,
            max_attempts=2,
        ),
    )
    journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="episode_controller",
        message_type="terminal",
        in_reply_to=repair.event_id,
        input_artifact_ids=[repair.event_id],
        payload=TerminalPayload(
            outcome="failure",
            reason="test stop",
            reward_total=-1.0,
            reward_components={"test": -1.0},
        ),
    )

    candidate = EpisodeRecords.load(journal.directory).sft_candidates()[0]

    assert candidate.output_event_id == proposal.event_id
    assert candidate.disposition == "accepted_upstream_repair"


@pytest.mark.parametrize(
    ("actor", "validator"),
    [
        ("single_control", "single_validator"),
        ("environment_generator", "environment_validator"),
    ],
)
def test_all_model_workflow_variants_produce_sft_candidates(
    tmp_path: Path,
    actor: str,
    validator: str,
) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id=f"episode-{actor}")
    trigger = journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent=actor,
        message_type="handoff",
        payload=SampleHandoff(label="request"),
    )
    _append_success_trace(
        journal=journal,
        trigger_event_id=trigger.event_id,
        agent_name=actor,
        value="accepted",
    )
    proposal = journal.append_event(
        round_index=0,
        source_agent=actor,
        target_agent="compiler",
        message_type="proposal",
        in_reply_to=trigger.event_id,
        input_artifact_ids=[trigger.event_id],
        payload=SampleDecision(value="accepted"),
    )
    accepted = journal.append_event(
        round_index=0,
        source_agent=validator,
        target_agent="compiler",
        message_type="validation_result",
        in_reply_to=proposal.event_id,
        input_artifact_ids=[proposal.event_id],
        payload=ValidationResultPayload(notes=["accepted"]),
    )
    journal.append_event(
        round_index=0,
        source_agent="episode_controller",
        target_agent="episode_controller",
        message_type="terminal",
        in_reply_to=accepted.event_id,
        input_artifact_ids=[accepted.event_id],
        payload=TerminalPayload(
            outcome="success",
            reason="done",
            reward_total=1.0,
            reward_components={"objective": 1.0},
        ),
    )

    candidates = EpisodeRecords.load(journal.directory).sft_candidates()

    assert len(candidates) == 1
    assert candidates[0].trace.agent_name == actor
