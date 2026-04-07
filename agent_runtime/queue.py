from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from .runtime_store import acquire_task_lease, complete_task, enqueue_task, get_task_status

from .artifacts import ArtifactEnvelope, ArtifactStore
from .workspace import runtime_root


class FileTaskQueue:
    def __init__(self, agent_name: str, *, lease_seconds: int = 60) -> None:
        self.agent_name = str(agent_name or "").strip()
        self.lease_seconds = max(1, int(lease_seconds))
        self.root = runtime_root() / "queues" / self.agent_name
        self.leases_dir = self.root / "leases"
        self.failures_dir = self.root / "failures"
        self.dead_letter_dir = self.root / "dead_letter"
        self.leases_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.dead_letter_dir.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore()

    def track_request(self, envelope: ArtifactEnvelope) -> None:
        enqueued = enqueue_task(
            artifact_id=envelope.artifact_id,
            artifact_type=envelope.artifact_type,
            source_agent=envelope.source_agent,
            target_agent=envelope.target_agent,
            session_id=envelope.session_id,
            snapshot_id=envelope.snapshot_id,
            correlation_id=envelope.correlation_id,
        )
        if not enqueued:
            raise RuntimeError(f"Failed to track queued task for artifact {envelope.artifact_id}.")

    def list_pending_requests(self) -> Iterable[Path]:
        request_root = self.store.request_dir("*", self.agent_name).parent.parent
        interfaces_root = self.store.root
        if not interfaces_root.exists():
            return []
        candidates: list[Path] = []
        for pair_dir in interfaces_root.glob(f"*__{self.agent_name}"):
            request_dir = pair_dir / "requests"
            if request_dir.exists():
                candidates.extend(sorted(request_dir.glob("*.json")))
        return sorted(candidates)

    def _lease_path(self, artifact_id: str) -> Path:
        return self.leases_dir / f"{artifact_id}.json"

    def _read_lease(self, lease_path: Path) -> Optional[dict]:
        if not lease_path.exists():
            return None
        try:
            return json.loads(lease_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _lease_is_active(self, payload: dict) -> bool:
        expires_at = float(payload.get("expires_at", 0.0) or 0.0)
        return expires_at > time.time()

    def claim_artifact(self, envelope: ArtifactEnvelope) -> bool:
        lease_path = self._lease_path(envelope.artifact_id)
        current = self._read_lease(lease_path)
        if current and self._lease_is_active(current):
            return False

        lease_payload = {
            "artifact_id": envelope.artifact_id,
            "target_agent": self.agent_name,
            "lease_owner": f"{self.agent_name}-{uuid4()}",
            "expires_at": time.time() + self.lease_seconds,
        }
        leased = acquire_task_lease(
            artifact_id=envelope.artifact_id,
            target_agent=self.agent_name,
            lease_owner=lease_payload["lease_owner"],
            lease_seconds=self.lease_seconds,
        )
        if not leased:
            return False
        tmp_path = lease_path.with_suffix(f".tmp-{uuid4().hex}")
        tmp_path.write_text(json.dumps(lease_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, lease_path)
        return True

    def mark_success(self, envelope: ArtifactEnvelope) -> None:
        lease_path = self._lease_path(envelope.artifact_id)
        if lease_path.exists():
            lease_path.unlink()
        completed = complete_task(artifact_id=envelope.artifact_id, target_agent=self.agent_name, status="succeeded")
        if not completed:
            raise RuntimeError(f"Failed to mark task success for artifact {envelope.artifact_id}.")

    def mark_failure(self, envelope: ArtifactEnvelope, error: Exception) -> None:
        lease_path = self._lease_path(envelope.artifact_id)
        if lease_path.exists():
            lease_path.unlink()
        completed = complete_task(
            artifact_id=envelope.artifact_id,
            target_agent=self.agent_name,
            status="failed",
            error=str(error),
        )
        if not completed:
            raise RuntimeError(f"Failed to mark task failure for artifact {envelope.artifact_id}.")
        status = get_task_status(envelope.artifact_id, self.agent_name)
        target_dir = self.dead_letter_dir if status == "dead_letter" else self.failures_dir
        target_path = target_dir / f"{envelope.artifact_id}.json"
        target_path.write_text(
            json.dumps(
                {
                    "artifact_id": envelope.artifact_id,
                    "error": str(error),
                    "target_agent": self.agent_name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
