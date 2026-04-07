from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.common import dataset_output_path, eval_dataset_dir, export_output_path
from sft_data.schemas import ChatmlSftRecord
from sft_data.tool_call.build_evalscope_call_decision_dataset import (
    DEFAULT_AGENT_NAME,
    EvalScopeToolCallDecisionRecord,
    build_evalscope_call_decision_records,
)


def _load_jsonl_models(path: Path, model_type: Type[BaseModel]) -> List[BaseModel]:
    if not path.exists():
        return []
    rows: List[BaseModel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(model_type.model_validate(payload))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _role_to_llama_factory(role: str) -> str:
    mapping = {
        "user": "human",
        "assistant": "gpt",
        "tool": "observation",
    }
    return mapping.get(role, role)


def _compact_chatml_metadata(record: ChatmlSftRecord, *, split: str) -> Dict[str, Any]:
    metadata = record.metadata
    return {
        "task": record.task,
        "agent": record.agent,
        "split": split,
        "trace_id": metadata.get("trace_id"),
        "session_id": metadata.get("session_id"),
        "snapshot_id": metadata.get("snapshot_id"),
        "tool_usage_valid": metadata.get("tool_usage_valid"),
        "failed_tool_names": list(metadata.get("failed_tool_names") or []),
        "semantic_judge_summary": metadata.get("semantic_judge_summary"),
    }


def chatml_record_to_llama_factory(record: ChatmlSftRecord, *, split: str) -> Dict[str, Any]:
    system_prompt = None
    conversations: List[Dict[str, str]] = []
    for index, message in enumerate(record.messages):
        if index == 0 and message.role == "system":
            system_prompt = message.content
            continue
        conversations.append(
            {
                "from": _role_to_llama_factory(message.role),
                "value": message.content,
            }
        )

    payload: Dict[str, Any] = {
        "id": record.sample_id,
        "conversations": conversations,
        "metadata": _compact_chatml_metadata(record, split=split),
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    return payload


def _tool_call_target(record: EvalScopeToolCallDecisionRecord) -> str:
    if not record.should_call_tool:
        return "<no_tool_call/>"

    tool_name = str(record.metadata.get("expected_tool_name") or "").strip()
    if not tool_name:
        raise ValueError(f"Record {record.sample_id} is missing expected_tool_name")
    tool_args = record.metadata.get("expected_tool_args") or {}
    if not isinstance(tool_args, dict):
        raise TypeError(f"Record {record.sample_id} expected_tool_args must be an object")
    return f"<tool_call name=\"{tool_name}\">{json.dumps(tool_args, ensure_ascii=False)}</tool_call>"


def tool_call_record_to_llama_factory(record: EvalScopeToolCallDecisionRecord) -> Dict[str, Any]:
    conversations = [
        {
            "from": _role_to_llama_factory(message.role),
            "value": message.content,
        }
        for message in record.messages
    ]
    conversations.append({"from": "gpt", "value": _tool_call_target(record)})
    return {
        "id": record.sample_id,
        "conversations": conversations,
        "tools": list(record.tools),
        "metadata": {
            "task": record.task,
            "agent": record.agent,
            "should_call_tool": record.should_call_tool,
            "boundary_type": record.metadata.get("boundary_type"),
            "expected_tool_name": record.metadata.get("expected_tool_name"),
            "expected_tool_args": record.metadata.get("expected_tool_args"),
            "rationale": record.metadata.get("rationale"),
        },
    }


def export_llama_factory_datasets(
    *,
    agent_name: str,
    chatml_success_input: Path,
    chatml_failure_input: Path,
    tool_call_input: Path | None,
    chatml_success_output: Path,
    chatml_failure_output: Path,
    tool_call_output: Path,
    manifest_output: Path,
) -> Dict[str, Any]:
    chatml_success_records = [
        chatml_record_to_llama_factory(record, split="success")
        for record in _load_jsonl_models(chatml_success_input, ChatmlSftRecord)
    ]
    chatml_failure_records = [
        chatml_record_to_llama_factory(record, split="failure")
        for record in _load_jsonl_models(chatml_failure_input, ChatmlSftRecord)
    ]

    if tool_call_input is not None and tool_call_input.exists():
        raw_tool_call_records = _load_jsonl_models(tool_call_input, EvalScopeToolCallDecisionRecord)
    else:
        raw_tool_call_records = build_evalscope_call_decision_records(agent_name=agent_name)
    tool_call_records = [
        tool_call_record_to_llama_factory(record)
        for record in raw_tool_call_records
    ]

    _write_json(chatml_success_output, chatml_success_records)
    _write_json(chatml_failure_output, chatml_failure_records)
    _write_json(tool_call_output, tool_call_records)

    manifest = {
        "agent": agent_name,
        "datasets": {
            "tool_call_decision": {
                "path": str(tool_call_output),
                "records": len(tool_call_records),
                "recommended_usage": "stage1_tool_call_warmup_sft",
            },
            "chatml_success": {
                "path": str(chatml_success_output),
                "records": len(chatml_success_records),
                "recommended_usage": "stage2_chatml_sft",
            },
            "chatml_failure": {
                "path": str(chatml_failure_output),
                "records": len(chatml_failure_records),
                "recommended_usage": "analysis_or_preference_mining_only",
            },
        },
        "recommended_pipeline": [
            "先用 tool_call_decision 做单步工具决策 warmup。",
            "再用 chatml_success 做多步 ChatML SFT。",
            "chatml_failure 默认不直接并入 SFT，而是用于误差分析、Judge 回归和后续偏好数据构造。",
        ],
    }
    _write_json(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export intent_encoding datasets into a Llama-Factory friendly JSON format.")
    parser.add_argument("--agent", type=str, default=DEFAULT_AGENT_NAME, help="Agent name used to scope input and output paths.")
    parser.add_argument(
        "--chatml-success-input",
        type=Path,
        default=None,
        help="Optional ChatML success JSONL input path.",
    )
    parser.add_argument(
        "--chatml-failure-input",
        type=Path,
        default=None,
        help="Optional ChatML failure JSONL input path.",
    )
    parser.add_argument(
        "--tool-call-input",
        type=Path,
        default=None,
        help="Optional tool-call decision JSONL input path. If omitted and the default file is absent, records are rebuilt in memory.",
    )
    parser.add_argument(
        "--chatml-success-output",
        type=Path,
        default=None,
        help="Output JSON path for ChatML success exports.",
    )
    parser.add_argument(
        "--chatml-failure-output",
        type=Path,
        default=None,
        help="Output JSON path for ChatML failure exports.",
    )
    parser.add_argument(
        "--tool-call-output",
        type=Path,
        default=None,
        help="Output JSON path for tool-call decision exports.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Output JSON path for the export manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chatml_success_input = args.chatml_success_input or dataset_output_path(args.agent, "chatml", "success", "iea_chatml_sft_v1.jsonl")
    chatml_failure_input = args.chatml_failure_input or dataset_output_path(args.agent, "chatml", "failure", "iea_chatml_sft_v1.jsonl")
    tool_call_input = args.tool_call_input or (eval_dataset_dir(args.agent, "evalscope") / "evalscope_call_decision_records_v1.jsonl")
    chatml_success_output = args.chatml_success_output or export_output_path(args.agent, "llama_factory", "chatml", "chatml_success_v1.json")
    chatml_failure_output = args.chatml_failure_output or export_output_path(args.agent, "llama_factory", "chatml", "chatml_failure_v1.json")
    tool_call_output = args.tool_call_output or export_output_path(args.agent, "llama_factory", "tool_call", "tool_call_decision_v1.json")
    manifest_output = args.manifest_output or export_output_path(args.agent, "llama_factory", "manifests", "dataset_manifest_v1.json")

    manifest = export_llama_factory_datasets(
        agent_name=args.agent,
        chatml_success_input=chatml_success_input,
        chatml_failure_input=chatml_failure_input,
        tool_call_input=tool_call_input,
        chatml_success_output=chatml_success_output,
        chatml_failure_output=chatml_failure_output,
        tool_call_output=tool_call_output,
        manifest_output=manifest_output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()