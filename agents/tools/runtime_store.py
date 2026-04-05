from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from database.models import (
    AgentArtifact,
    AgentHandoffRecord,
    AgentTask,
    EpisodicExperience,
    SessionContext,
    SessionStageResult,
)
from agents.tools.db_tool import session_scope
from utils.logger import setup_logger


logger = setup_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _summary(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        keys = sorted(payload.keys())[:20]
        return {"keys": keys}
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump(mode="json")
        if isinstance(dumped, dict):
            return _summary(dumped)
    return {"type": type(payload).__name__}


def create_or_update_session(
    session_id: str,
    *,
    status: Optional[str] = None,
    current_stage: Optional[str] = None,
    current_snapshot_id: Optional[str] = None,
    current_artifact_id: Optional[str] = None,
    round_index: Optional[int] = None,
    last_error: Optional[str] = None,
) -> bool:
    if not session_id:
        return False

    try:
        with session_scope() as session:
            row = session.query(SessionContext).filter(SessionContext.session_id == session_id).first()
            if row is None:
                row = SessionContext(session_id=session_id)
                session.add(row)
            if status is not None:
                row.status = status
            if current_stage is not None:
                row.current_stage = current_stage
                row.current_step = current_stage
            if current_snapshot_id is not None:
                row.current_snapshot_id = current_snapshot_id
            if current_artifact_id is not None:
                row.current_artifact_id = current_artifact_id
            if round_index is not None:
                row.round_index = int(round_index)
            if last_error is not None:
                row.last_error = str(last_error)
        return True
    except Exception as exc:
        logger.warning("Failed to upsert session control row %s: %s", session_id, exc)
        return False


def record_artifact(
    *,
    artifact_id: str,
    correlation_id: str,
    session_id: str,
    snapshot_id: str,
    source_agent: str,
    target_agent: str,
    artifact_type: str,
    kind: str,
    path: str,
    payload: Any,
) -> bool:
    if not artifact_id:
        return False
    try:
        with session_scope() as session:
            row = session.query(AgentArtifact).filter(AgentArtifact.artifact_id == artifact_id).first()
            if row is None:
                row = AgentArtifact(artifact_id=artifact_id)
                session.add(row)
            row.correlation_id = correlation_id or None
            row.session_id = session_id or None
            row.snapshot_id = snapshot_id or None
            row.source_agent = source_agent
            row.target_agent = target_agent
            row.artifact_type = artifact_type
            row.kind = kind
            row.path = path
            row.payload_summary = _summary(payload)
        return True
    except Exception as exc:
        logger.warning("Failed to record artifact %s: %s", artifact_id, exc)
        return False


def enqueue_task(
    *,
    artifact_id: str,
    artifact_type: str,
    source_agent: str,
    target_agent: str,
    session_id: str,
    snapshot_id: str,
    correlation_id: str,
    max_attempts: int = 3,
) -> bool:
    if not artifact_id or not target_agent:
        return False
    try:
        with session_scope() as session:
            row = (
                session.query(AgentTask)
                .filter(AgentTask.artifact_id == artifact_id, AgentTask.target_agent == target_agent)
                .first()
            )
            if row is None:
                row = AgentTask(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    source_agent=source_agent,
                    target_agent=target_agent,
                    session_id=session_id or None,
                    snapshot_id=snapshot_id or None,
                    correlation_id=correlation_id or None,
                    max_attempts=max(1, int(max_attempts)),
                )
                session.add(row)
            row.status = "queued"
            row.last_error = None
        return True
    except Exception as exc:
        logger.warning("Failed to enqueue task for artifact %s: %s", artifact_id, exc)
        return False


def acquire_task_lease(
    *,
    artifact_id: str,
    target_agent: str,
    lease_owner: str,
    lease_seconds: int = 60,
) -> bool:
    if not artifact_id or not target_agent or not lease_owner:
        return False
    now = _utcnow()
    try:
        with session_scope() as session:
            row = (
                session.query(AgentTask)
                .filter(AgentTask.artifact_id == artifact_id, AgentTask.target_agent == target_agent)
                .first()
            )
            if row is None:
                return False
            if row.status in {"succeeded", "dead_letter"}:
                return False
            if row.lease_expires_at and row.lease_expires_at > now and row.lease_owner and row.lease_owner != lease_owner:
                return False
            row.status = "running"
            row.lease_owner = lease_owner
            row.lease_expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
            row.attempts = int(row.attempts or 0) + 1
        return True
    except Exception as exc:
        logger.warning("Failed to lease task %s/%s: %s", target_agent, artifact_id, exc)
        return False


def complete_task(
    *,
    artifact_id: str,
    target_agent: str,
    status: str,
    error: str = "",
) -> bool:
    try:
        with session_scope() as session:
            row = (
                session.query(AgentTask)
                .filter(AgentTask.artifact_id == artifact_id, AgentTask.target_agent == target_agent)
                .first()
            )
            if row is None:
                return False
            normalized = str(status or "").strip().lower()
            if normalized == "failed" and int(row.attempts or 0) >= int(row.max_attempts or 3):
                row.status = "dead_letter"
            else:
                row.status = normalized
            row.last_error = str(error or "") or None
            row.lease_owner = None
            row.lease_expires_at = None
        return True
    except Exception as exc:
        logger.warning("Failed to complete task %s/%s: %s", target_agent, artifact_id, exc)
        return False


def get_task_status(artifact_id: str, target_agent: str) -> Optional[str]:
    try:
        with session_scope() as session:
            row = (
                session.query(AgentTask)
                .filter(AgentTask.artifact_id == artifact_id, AgentTask.target_agent == target_agent)
                .first()
            )
            return None if row is None else str(row.status or "")
    except Exception as exc:
        logger.warning("Failed to get task status %s/%s: %s", target_agent, artifact_id, exc)
        return None


def record_handoff(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    source_agent: str,
    target_agent: str,
    artifact_id: str,
    artifact_type: str,
    summary: str,
    payload: Any,
) -> bool:
    try:
        with session_scope() as session:
            session.add(
                AgentHandoffRecord(
                    session_id=session_id or None,
                    snapshot_id=snapshot_id or None,
                    round_index=int(round_index or 0),
                    source_agent=source_agent,
                    target_agent=target_agent,
                    artifact_id=artifact_id or None,
                    artifact_type=artifact_type,
                    summary=summary or None,
                    handoff_payload=payload if isinstance(payload, dict) else _summary(payload),
                )
            )
        return True
    except Exception as exc:
        logger.warning("Failed to record handoff %s -> %s: %s", source_agent, target_agent, exc)
        return False


def record_stage_result(
    *,
    session_id: str,
    snapshot_id: str,
    round_index: int,
    stage_name: str,
    artifact_id: str,
    status: str,
    payload: Any,
) -> bool:
    try:
        with session_scope() as session:
            session.add(
                SessionStageResult(
                    session_id=session_id,
                    snapshot_id=snapshot_id or None,
                    round_index=int(round_index or 0),
                    stage_name=stage_name,
                    artifact_id=artifact_id or None,
                    status=status,
                    payload=payload if isinstance(payload, dict) else _summary(payload),
                )
            )
        return True
    except Exception as exc:
        logger.warning("Failed to record stage result %s: %s", stage_name, exc)
        return False


def list_stage_results(session_id: str) -> List[Dict[str, Any]]:
    try:
        with session_scope() as session:
            rows = (
                session.query(SessionStageResult)
                .filter(SessionStageResult.session_id == session_id)
                .order_by(SessionStageResult.created_at.asc(), SessionStageResult.id.asc())
                .all()
            )
            return [
                {
                    "stage_name": row.stage_name,
                    "status": row.status,
                    "artifact_id": row.artifact_id,
                    "payload": row.payload,
                    "round_index": row.round_index,
                }
                for row in rows
            ]
    except Exception as exc:
        logger.warning("Failed to list stage results for %s: %s", session_id, exc)
        return []


def record_episodic_experience(
    *,
    raw_intent: str,
    applied_policy: Dict[str, Any],
    environment_state: Dict[str, Any],
    feedback_metrics: Dict[str, Any],
    reward_score: float,
    intent_vector: Optional[List[float]] = None,
) -> bool:
    try:
        with session_scope() as session:
            session.add(
                EpisodicExperience(
                    raw_intent=raw_intent,
                    applied_policy=applied_policy,
                    environment_state=environment_state,
                    feedback_metrics=feedback_metrics,
                    reward_score=float(reward_score),
                    intent_vector=intent_vector,
                )
            )
        return True
    except Exception as exc:
        logger.warning("Failed to record episodic experience: %s", exc)
        return False
