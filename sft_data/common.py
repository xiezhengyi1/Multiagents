from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel


_VALID_DATASET_SPLITS = {"success", "failure"}


@dataclass(frozen=True)
class ArtifactPair:
    request: Dict[str, Any]
    response: Dict[str, Any]
    matched_by: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def agent_root(agent_name: str) -> Path:
    return repo_root() / "sft_data" / agent_name


def raw_trace_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "raw_traces"


def trace_file(agent_name: str) -> Path:
    return raw_trace_dir(agent_name) / f"{agent_name}.jsonl"


def datasets_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "datasets"


def dataset_dir(agent_name: str, dataset_name: str) -> Path:
    return datasets_dir(agent_name) / dataset_name


def dataset_split_dir(agent_name: str, dataset_name: str, split: str) -> Path:
    if split not in _VALID_DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset split: {split}")
    return dataset_dir(agent_name, dataset_name) / split


def dataset_output_path(agent_name: str, dataset_name: str, split: str, filename: str) -> Path:
    return dataset_split_dir(agent_name, dataset_name, split) / filename


def exports_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "exports"


def export_format_dir(agent_name: str, export_format: str) -> Path:
    return exports_dir(agent_name) / export_format


def export_dataset_dir(agent_name: str, export_format: str, dataset_name: str) -> Path:
    return export_format_dir(agent_name, export_format) / dataset_name


def export_output_path(agent_name: str, export_format: str, dataset_name: str, filename: str) -> Path:
    return export_dataset_dir(agent_name, export_format, dataset_name) / filename


def evals_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "evals"


def evaluator_dir(agent_name: str, evaluator_name: str) -> Path:
    return evals_dir(agent_name) / evaluator_name


def eval_dataset_dir(agent_name: str, evaluator_name: str) -> Path:
    return evaluator_dir(agent_name, evaluator_name) / "dataset"


def eval_runs_dir(agent_name: str, evaluator_name: str) -> Path:
    return evaluator_dir(agent_name, evaluator_name) / "runs"


def eval_run_dir(agent_name: str, evaluator_name: str, run_name: str) -> Path:
    return eval_runs_dir(agent_name, evaluator_name) / run_name


def processed_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "processed"


def rejects_dir(agent_name: str) -> Path:
    return agent_root(agent_name) / "rejects"


def ensure_agent_layout(agent_name: str) -> None:
    agent_root(agent_name).mkdir(parents=True, exist_ok=True)
    raw_trace_dir(agent_name).mkdir(parents=True, exist_ok=True)
    datasets_dir(agent_name).mkdir(parents=True, exist_ok=True)
    exports_dir(agent_name).mkdir(parents=True, exist_ok=True)
    evals_dir(agent_name).mkdir(parents=True, exist_ok=True)
    processed_dir(agent_name).mkdir(parents=True, exist_ok=True)
    rejects_dir(agent_name).mkdir(parents=True, exist_ok=True)


def artifact_dir(project_root: Path, relative: str) -> Path:
    path = project_root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_envelopes(path: Path) -> List[Dict[str, Any]]:
    envelopes: List[Dict[str, Any]] = []
    for artifact_path in sorted(path.glob("*.json")):
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{artifact_path} must contain a JSON object")
        payload["_path"] = str(artifact_path)
        envelopes.append(payload)
    return envelopes


def find_request_for_response(response: Dict[str, Any], requests: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any] | None, str | None]:
    correlation_id = str(response.get("correlation_id") or "").strip()
    correlation_matches = [req for req in requests if str(req.get("correlation_id") or "").strip() == correlation_id]
    if len(correlation_matches) == 1:
        return correlation_matches[0], "correlation_id"

    upstream_ids = response.get("upstream_artifact_ids", [])
    if not isinstance(upstream_ids, list):
        raise TypeError("upstream_artifact_ids must be a list")
    upstream_matches = [req for req in requests if str(req.get("artifact_id") or "") in upstream_ids]
    if len(upstream_matches) == 1:
        return upstream_matches[0], "upstream_artifact_ids"
    return None, None


def load_artifact_pairs(
    project_root: Path,
    *,
    request_relative: str,
    response_relative: str,
) -> Tuple[List[ArtifactPair], List[Dict[str, Any]], int]:
    request_dir = artifact_dir(project_root, request_relative)
    response_dir = artifact_dir(project_root, response_relative)

    requests = load_envelopes(request_dir)
    responses = load_envelopes(response_dir)
    pairs: List[ArtifactPair] = []
    rejects: List[Dict[str, Any]] = []

    for response in responses:
        matched_request, matched_by = find_request_for_response(response, requests)
        if matched_request is None or matched_by is None:
            rejects.append(
                {
                    "kind": "pairing_failed",
                    "artifact_id": response.get("artifact_id"),
                    "reason": "no request matched by correlation_id or upstream_artifact_ids",
                }
            )
            continue
        pairs.append(ArtifactPair(request=matched_request, response=response, matched_by=matched_by))

    return pairs, rejects, len(requests) + len(responses)


def load_trace_records(trace_file: Path, trace_model: Type[BaseModel]) -> List[BaseModel]:
    if not trace_file.exists():
        return []
    text = trace_file.read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = []

    # 兼容历史 trace 文件中因缺少换行导致的多对象粘连。
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break

        row_line_number = text.count("\n", 0, position) + 1
        try:
            payload, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            error_line_number = text.count("\n", 0, exc.pos) + 1
            raise ValueError(f"{trace_file}:{error_line_number} invalid JSON trace payload: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"{trace_file}:{row_line_number} JSONL row must be an object")
        rows.append(payload)

    return [trace_model.model_validate(row) for row in rows]
