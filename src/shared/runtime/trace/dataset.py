from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

from shared.contracts import (
    TraceDatasetManifest,
    TraceSegmentManifest,
    TraceWorkerManifest,
    utc_now,
)
from shared.runtime.io.paths import (
    current_trace_identity,
    trace_run_root,
    trace_runs_root,
    trace_worker_root,
)


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def segment_manifest_path(segment_path: Path) -> Path:
    return segment_path.with_suffix(".manifest.json")


def open_trace_segment(
    *,
    segment_path: Path,
    dataset_run_id: str,
    worker_id: str,
    agent_name: str,
    root: Path | None = None,
) -> TraceSegmentManifest:
    base = (Path(root) if root is not None else trace_runs_root()).resolve()
    segment_path = segment_path.resolve()
    manifest_path = segment_manifest_path(segment_path)
    if manifest_path.exists():
        manifest = TraceSegmentManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.status != "writing":
            raise RuntimeError(f"finalized trace segment cannot be reopened: {segment_path}")
        expected_identity = (dataset_run_id, worker_id, agent_name)
        actual_identity = (
            manifest.dataset_run_id,
            manifest.worker_id,
            manifest.agent_name,
        )
        if actual_identity != expected_identity:
            raise RuntimeError(f"trace segment identity mismatch: {segment_path}")
        if not segment_path.is_file():
            raise RuntimeError(f"trace segment manifest has no JSONL segment: {segment_path}")
        if _jsonl_record_count(segment_path) != manifest.record_count:
            raise RuntimeError(f"trace segment record_count mismatch: {segment_path}")
        return manifest
    if segment_path.exists():
        raise RuntimeError(f"unmanifested trace segment already exists: {segment_path}")
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.touch()
    now = utc_now()
    manifest = TraceSegmentManifest(
        dataset_run_id=dataset_run_id,
        worker_id=worker_id,
        agent_name=agent_name,
        relative_path=segment_path.relative_to(base).as_posix(),
        status="writing",
        created_at=now,
        updated_at=now,
        record_count=0,
    )
    _write_model_atomic(manifest_path, manifest)
    return manifest


def record_trace_append(segment_path: Path, manifest: TraceSegmentManifest) -> TraceSegmentManifest:
    if manifest.status != "writing":
        raise RuntimeError("cannot append to a finalized trace segment")
    updated = manifest.model_copy(
        update={
            "updated_at": utc_now(),
            "record_count": manifest.record_count + 1,
        }
    )
    _write_model_atomic(segment_manifest_path(segment_path), updated)
    return updated


def _finalize_segment(
    manifest_path: Path,
    *,
    root: Path | None = None,
) -> TraceSegmentManifest:
    manifest = TraceSegmentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.status == "finalized":
        return manifest
    run_root = trace_run_root(manifest.dataset_run_id, root=root)
    segment_path = (run_root.parent / manifest.relative_path).resolve()
    if not segment_path.is_file():
        raise RuntimeError(f"trace segment is missing: {segment_path}")
    actual_count = _jsonl_record_count(segment_path)
    if actual_count != manifest.record_count:
        raise RuntimeError(f"trace segment record_count mismatch: {segment_path}")
    finalized = manifest.model_copy(
        update={
            "status": "finalized",
            "updated_at": utc_now(),
            "sha256": _sha256(segment_path),
        }
    )
    _write_model_atomic(manifest_path, finalized)
    return finalized


def finalize_trace_worker(
    *,
    dataset_run_id: str | None = None,
    worker_id: str | None = None,
    root: Path | None = None,
) -> Path:
    current_run_id, current_worker_id = current_trace_identity()
    run_id = dataset_run_id or current_run_id
    resolved_worker_id = worker_id or current_worker_id
    worker_root = trace_worker_root(run_id, resolved_worker_id, root=root)
    segment_manifest_paths = sorted(worker_root.glob("*/*.manifest.json"))
    if not segment_manifest_paths:
        raise RuntimeError(f"trace worker has no segments: {worker_root}")
    segments = [_finalize_segment(path, root=root) for path in segment_manifest_paths]
    manifest = TraceWorkerManifest(
        dataset_run_id=run_id,
        worker_id=resolved_worker_id,
        finalized_at=utc_now(),
        record_count=sum(segment.record_count for segment in segments),
        segments=segments,
    )
    output = worker_root / "manifest.json"
    _write_model_atomic(output, manifest)
    return output


def finalize_trace_dataset_run(dataset_run_id: str, *, root: Path | None = None) -> Path:
    run_root = trace_run_root(dataset_run_id, root=root)
    worker_manifest_paths = sorted((run_root / "workers").glob("*/manifest.json"))
    if not worker_manifest_paths:
        raise RuntimeError(f"trace dataset run has no finalized workers: {run_root}")
    workers = [
        TraceWorkerManifest.model_validate_json(path.read_text(encoding="utf-8"))
        for path in worker_manifest_paths
    ]
    manifest = TraceDatasetManifest(
        dataset_run_id=dataset_run_id,
        finalized_at=utc_now(),
        record_count=sum(worker.record_count for worker in workers),
        workers=workers,
    )
    output = run_root / "manifest.json"
    _write_model_atomic(output, manifest)
    return output


__all__ = [
    "finalize_trace_dataset_run",
    "finalize_trace_worker",
    "open_trace_segment",
    "record_trace_append",
    "segment_manifest_path",
]
