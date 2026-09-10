from datetime import datetime, timezone

from control_runtime.actors.common import ControlRequest
from control_runtime.graph import ContextBuilder
from control_runtime.simulation.contracts import EnvironmentObservation, ObservationRequest
from shared.contracts.coordination import (
    ProposalPayload,
    RepairRequestPayload,
    ValidationIssue,
)
from shared.runtime.journal import EpisodeJournal


class SampleProposal(ProposalPayload):
    value: str


def test_context_keeps_observations_from_invalidated_proposal(tmp_path) -> None:
    journal = EpisodeJournal(root=tmp_path, episode_id="episode")
    request = journal.append_event(
        round_index=0,
        source_agent="user",
        target_agent="main",
        message_type="handoff",
        payload=ControlRequest(user_input="protect flow"),
    )
    observation = journal.append_event(
        round_index=0,
        source_agent="simulator_adapter",
        target_agent="sba",
        message_type="observation",
        in_reply_to=request.event_id,
        input_artifact_ids=[request.event_id],
        payload=EnvironmentObservation(
            base_snapshot_id="snapshot",
            observed_at=datetime.now(timezone.utc),
            query=ObservationRequest(
                query_name="get_sm_ue_flow_catalog",
                arguments={"supi": "imsi-208930000000001"},
            ),
            result={"flow_catalog": []},
        ),
    )
    rejected = journal.append_event(
        round_index=0,
        source_agent="sba",
        target_agent="sba_validator",
        message_type="proposal",
        in_reply_to=request.event_id,
        input_artifact_ids=[request.event_id, observation.event_id],
        payload=SampleProposal(value="rejected proposal"),
    )
    repair = journal.append_event(
        round_index=0,
        source_agent="sba_validator",
        target_agent="sba",
        message_type="repair_request",
        in_reply_to=rejected.event_id,
        invalidated_artifact_ids=[rejected.event_id],
        payload=RepairRequestPayload(
            issues=[ValidationIssue(code="missing_subscription", message="missing")],
            attempt=1,
            max_attempts=1,
        ),
    )

    view, _ = ContextBuilder(
        journal=journal,
        token_budget=10_000,
        count_tokens=len,
    ).build(repair)

    assert view.source_event_ids == [request.event_id, observation.event_id, repair.event_id]
