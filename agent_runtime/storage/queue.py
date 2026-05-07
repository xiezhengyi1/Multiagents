from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from agent_runtime.io.files import read_json_file, write_json_file_atomic
from agent_runtime.io.paths import queue_root

from .artifacts import ArtifactEnvelope, ArtifactStore
from .runtime_store import acquire_task_lease, complete_task, get_task_status


class FileTaskQueue:
    def __init__(
        self,
        agent_name: str,
        *,
        lease_seconds: int = 60,
        root: Path | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.agent_name = str(agent_name or "").strip()
        self.lease_seconds = max(1, int(lease_seconds))
        self.root = Path(root) if root is not None else queue_root(self.agent_name)
        self.leases_dir = self.root / "leases"
        self.failures_dir = self.root / "failures"
        self.dead_letter_dir = self.root / "dead_letter"
        self.leases_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.dead_letter_dir.mkdir(parents=True, exist_ok=True)
        self.store = artifact_store or ArtifactStore()

    def list_pending_requests(self) -> Iterable[Path]:
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
            return read_json_file(lease_path)
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
        write_json_file_atomic(lease_path, lease_payload)
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
        write_json_file_atomic(
            target_path,
            {
                "artifact_id": envelope.artifact_id,
                "error": str(error),
                "target_agent": self.agent_name,
            },
        )
