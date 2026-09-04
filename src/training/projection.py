from __future__ import annotations

from shared.contracts.coordination import TerminalPayload

from .records import EpisodeRecords
from .schemas import EpisodeMetrics, RlTrajectory, SftExample


def project_sft_examples(records: EpisodeRecords) -> list[SftExample]:
    episode_id = _episode_id(records)
    examples: list[SftExample] = []
    for candidate in records.sft_candidates():
        output = candidate.trace.output_message
        if output is None:
            raise ValueError(
                f"successful trace {candidate.trace.trace_id} has no output"
            )
        examples.append(
            SftExample(
                example_id=candidate.trace.trace_id,
                episode_id=episode_id,
                actor=candidate.trace.agent_name,
                input_messages=candidate.trace.input_messages,
                output_message=output,
                tools=candidate.trace.tools,
                output_event_id=candidate.output_event_id,
                validation_event_id=candidate.validation_event_id,
                disposition=candidate.disposition,
            )
        )
    return examples


def project_rl_trajectory(records: EpisodeRecords) -> RlTrajectory:
    terminal_event = records.events[-1]
    return RlTrajectory(
        episode_id=_episode_id(records),
        events=records.events,
        traces=records.traces,
        terminal=TerminalPayload.model_validate(terminal_event["payload"]),
        metrics=project_episode_metrics(records),
    )


def project_episode_metrics(records: EpisodeRecords) -> EpisodeMetrics:
    terminal = TerminalPayload.model_validate(records.events[-1]["payload"])
    round_indices: list[int] = []
    repair_count = 0
    execution_attempt_count = 0
    conflict_count = 0
    objective_violation_count = 0
    execution_failure_count = 0
    for event in records.events:
        scope = event.get("scope")
        if not isinstance(scope, dict) or not isinstance(scope.get("round_index"), int):
            raise TypeError("every event scope requires integer round_index")
        round_indices.append(scope["round_index"])
        message_type = event.get("message_type")
        if message_type == "repair_request":
            repair_count += 1
        elif message_type == "execution_feedback":
            execution_attempt_count += 1
        elif message_type == "diagnosis":
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise TypeError("diagnosis payload must be an object")
            status = payload.get("status")
            if status == "blocked":
                conflicts = payload.get("conflicts")
                if not isinstance(conflicts, list):
                    raise TypeError("blocked conflict diagnosis requires conflicts")
                conflict_count += len(conflicts)
            elif status == "objective_violated":
                objective_violation_count += 1
            elif status == "execution_failed":
                execution_failure_count += 1
    return EpisodeMetrics(
        round_count=max(round_indices) + 1,
        repair_count=repair_count,
        execution_attempt_count=execution_attempt_count,
        model_call_count=len(records.traces),
        model_error_count=sum(trace.status == "error" for trace in records.traces),
        conflict_count=conflict_count,
        objective_violation_count=objective_violation_count,
        execution_failure_count=execution_failure_count,
        first_attempt_success=(
            terminal.outcome == "success"
            and execution_attempt_count == 1
            and repair_count == 0
        ),
        recovered_after_repair=(terminal.outcome == "success" and repair_count > 0),
    )


def _episode_id(records: EpisodeRecords) -> str:
    first_scope = records.events[0].get("scope")
    if not isinstance(first_scope, dict):
        raise TypeError("first event scope must be an object")
    episode_id = first_scope.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("first event scope requires episode_id")
    return episode_id


__all__ = [
    "project_episode_metrics",
    "project_rl_trajectory",
    "project_sft_examples",
]
