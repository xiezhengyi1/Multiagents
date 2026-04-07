from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_runtime import ArtifactEnvelope
from agents.intent_encoding.agent import IntentEncodingAgent
from generate_user_inputs import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_SUCCESS_RATE,
    build_output,
    load_network_state,
)
from sft_data.common import dataset_dir, dataset_output_path, load_trace_records
from sft_data.intent_encoding.build_iea_chatml_sft_dataset import _count_chatml_build_errors, split_iea_chatml_records, trace_to_iea_chatml_record
from sft_data.schemas import BuildReport, ChatmlSftRecord, MinimalTraceRecord, load_build_report, save_build_report, write_jsonl


class IntentEncodingRunner(Protocol):
    def handle_artifact(self, envelope: ArtifactEnvelope) -> Any:
        ...


def _trace_file_path(project_root: Path) -> Path:
    return project_root / "sft_data" / "intent_encoding" / "raw_traces" / "intent_encoding.jsonl"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(payload)
    return rows


def _normalize_user_input_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    messages = record.get("messages")
    if messages not in (None, ""):
        if not isinstance(messages, list):
            raise TypeError(f"Record #{index} field 'messages' must be a list")
        normalized_messages: List[Dict[str, str]] = []
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise TypeError(f"Record #{index} message #{message_index} must be an object")
            role = str(message.get("role") or "").strip().lower() or "user"
            content = str(message.get("content") or "")
            if not content.strip():
                continue
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages:
            raise ValueError(f"Record #{index} field 'messages' must contain at least one non-empty message")
        if normalized_messages[-1]["role"] != "user":
            raise ValueError(f"Record #{index} messages must end with a user role")
        user_input = normalized_messages[-1]["content"]
    else:
        normalized_messages = []
        user_input = record.get("user_input")
        if user_input in (None, ""):
            user_input = record.get("userInput")
        if user_input in (None, ""):
            raise ValueError(f"Record #{index} is missing 'user_input'/'userInput'")
        normalized_messages = [{"role": "user", "content": str(user_input)}]

    context = record.get("context")
    if context in (None, ""):
        context = record.get("conversation_context", "")

    return {
        **record,
        "user_input": str(user_input),
        "messages": normalized_messages,
        "context": str(context or ""),
    }


def load_user_input_records(
    *,
    user_inputs_path: Path | None,
    network_input_path: Path | None,
    count: int,
    target_success_rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if user_inputs_path is not None:
        if not user_inputs_path.exists():
            raise FileNotFoundError(user_inputs_path)
        if user_inputs_path.suffix.lower() == ".jsonl":
            payload: Any = _read_jsonl(user_inputs_path)
        else:
            payload = _read_json(user_inputs_path)

        if isinstance(payload, dict):
            records = payload.get("records")
            if not isinstance(records, list):
                raise TypeError(f"{user_inputs_path} must contain a top-level 'records' list")
        elif isinstance(payload, list):
            records = payload
        else:
            raise TypeError(f"{user_inputs_path} must contain a JSON object or array")

        return [_normalize_user_input_record(record, index) for index, record in enumerate(records, start=1)]

    network_state = load_network_state(network_input_path)
    payload = build_output(
        network_state,
        count=count,
        target_success_rate=target_success_rate,
        seed=seed,
    )
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError("Generated user input payload must contain a 'records' list")
    return [_normalize_user_input_record(record, index) for index, record in enumerate(records, start=1)]


def _write_envelope(path: Path, envelope: ArtifactEnvelope) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_latest_trace_for_session(project_root: Path, session_id: str) -> MinimalTraceRecord:
    traces = load_trace_records(_trace_file_path(project_root), MinimalTraceRecord)
    matches = [trace for trace in traces if trace.session_id == session_id]
    if not matches:
        raise ValueError(f"No trace found for session_id={session_id}")
    return matches[-1]


def _build_tool_usage_failure_reason(record: ChatmlSftRecord) -> str:
    semantic_validation = record.metadata.get("semantic_tool_usage_report") or {}
    failed_tools = list(record.metadata.get("failed_tools") or [])
    if failed_tools:
        details = []
        for item in failed_tools:
            tool_name = str(item.get("tool_name") or "unknown_tool")
            tool_call_id = str(item.get("tool_call_id") or "")
            issues = "; ".join(str(issue) for issue in item.get("issues") or [])
            label = f"{tool_name}({tool_call_id})" if tool_call_id else tool_name
            details.append(f"{label}: {issues}" if issues else label)
        suffix = ""
        semantic_summary = str(semantic_validation.get("summary") or "").strip()
        if semantic_summary:
            suffix = f"; semantic summary: {semantic_summary}"
        return "trace failed intent encoding tool usage validation; failed tools: " + " | ".join(details) + suffix

    trace_errors = list((record.metadata.get("tool_usage_report") or {}).get("trace_errors") or [])
    if trace_errors:
        return "trace failed intent encoding tool usage validation; trace errors: " + " | ".join(str(item) for item in trace_errors)

    return "trace failed intent encoding tool usage validation"


def build_chatml_records_from_user_inputs(
    *,
    agent: IntentEncodingRunner,
    records: Sequence[Dict[str, Any]],
    session_prefix: str,
    snapshot_id: str,
    default_context: str,
    project_root: Path,
    allow_user_interaction: bool = False,
    enable_semantic_judge: bool | None = None,
    semantic_judge: Any = None,
) -> Tuple[List[ChatmlSftRecord], List[Dict[str, Any]]]:
    chatml_records: List[ChatmlSftRecord] = []
    rejects: List[Dict[str, Any]] = []

    total = len(records)
    for index, record in enumerate(records, start=1):
        normalized_record = _normalize_user_input_record(record, index)
        user_input = str(normalized_record.get("user_input") or "").strip()
        if not user_input:
            raise ValueError(f"Record #{index} has empty user_input")

        context = str(normalized_record.get("context") or default_context or "")
        messages = normalized_record.get("messages") or [{"role": "user", "content": user_input}]
        session_id = str(normalized_record.get("session_id") or f"{session_prefix}-{index:05d}")
        record_snapshot_id = str(normalized_record.get("snapshot_id") or snapshot_id)

        preview = user_input.replace("\n", " ")[:120]
        print(f"[{index}/{total}] IEA ChatML start: {preview}", flush=True)

        request_envelope = ArtifactEnvelope(
            artifact_type="OperationIntentRequest",
            source_agent="coordinator",
            target_agent="intent_encoding",
            session_id=session_id,
            snapshot_id=record_snapshot_id,
            payload={
                "user_input": user_input,
                "context": context,
                "messages": messages,
                "allow_user_interaction": bool(allow_user_interaction),
            },
        )

        try:
            agent.handle_artifact(request_envelope)
            trace = _extract_latest_trace_for_session(project_root, session_id)
            record = trace_to_iea_chatml_record(
                trace,
                enable_semantic_judge=enable_semantic_judge,
                semantic_judge=semantic_judge,
            )
            chatml_records.append(record)
            if not bool(record.metadata.get("tool_usage_valid")):
                rejects.append(
                    {
                        "kind": "iea_tool_usage_validation_failed",
                        "record_index": index,
                        "session_id": session_id,
                        "user_input": user_input,
                        "reason": _build_tool_usage_failure_reason(record),
                        "failed_tools": record.metadata.get("failed_tools", []),
                        "semantic_tool_usage_report": record.metadata.get("semantic_tool_usage_report"),
                    }
                )
            print(f"[{index}/{total}] IEA ChatML done: session_id={session_id}", flush=True)
        except Exception as exc:
            rejects.append(
                {
                    "kind": "iea_chatml_generation_failed",
                    "record_index": index,
                    "session_id": session_id,
                    "user_input": user_input,
                    "reason": str(exc),
                }
            )
            print(f"[{index}/{total}] IEA ChatML reject: session_id={session_id} reason={exc}", flush=True)

    return chatml_records, rejects


def materialize_iea_artifacts(
    *,
    agent: IntentEncodingRunner,
    records: Sequence[Dict[str, Any]],
    staging_root: Path,
    session_prefix: str,
    snapshot_id: str,
    default_context: str,
    allow_user_interaction: bool = False,
) -> int:
    request_dir = staging_root / "runtime" / "interfaces" / "coordinator__intent_encoding" / "requests"
    response_dir = staging_root / "runtime" / "interfaces" / "intent_encoding__coordinator" / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    total = len(records)
    for index, record in enumerate(records, start=1):
        normalized_record = _normalize_user_input_record(record, index)
        user_input = str(normalized_record.get("user_input") or "").strip()
        if not user_input:
            raise ValueError(f"Record #{index} has empty user_input")
        context = str(normalized_record.get("context") or default_context or "")
        messages = normalized_record.get("messages") or [{"role": "user", "content": user_input}]
        session_id = str(normalized_record.get("session_id") or f"{session_prefix}-{index:05d}")
        record_snapshot_id = str(normalized_record.get("snapshot_id") or snapshot_id)

        preview = user_input.replace("\n", " ")[:120]
        print(f"[{index}/{total}] IEA start: {preview}", flush=True)

        request_envelope = ArtifactEnvelope(
            artifact_type="OperationIntentRequest",
            source_agent="coordinator",
            target_agent="intent_encoding",
            session_id=session_id,
            snapshot_id=record_snapshot_id,
            payload={
                "user_input": user_input,
                "context": context,
                "messages": messages,
                "allow_user_interaction": bool(allow_user_interaction),
            },
        )
        _write_envelope(request_dir / f"{request_envelope.artifact_id}.json", request_envelope)

        operation_intent = agent.handle_artifact(request_envelope)
        response_envelope = ArtifactEnvelope(
            artifact_type="OperationIntent",
            source_agent="intent_encoding",
            target_agent="coordinator",
            session_id=request_envelope.session_id,
            snapshot_id=request_envelope.snapshot_id,
            correlation_id=request_envelope.correlation_id,
            upstream_artifact_ids=[request_envelope.artifact_id],
            payload=operation_intent.model_dump(mode="json"),
        )
        _write_envelope(response_dir / f"{response_envelope.artifact_id}.json", response_envelope)
        print(
            f"[{index}/{total}] IEA done: session_id={request_envelope.session_id} correlation_id={request_envelope.correlation_id}",
            flush=True,
        )
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read generated user inputs, call the IEA agent, and build the IEA SFT dataset.",
    )
    parser.add_argument(
        "--user-inputs",
        type=Path,
        default=None,
        help="Path to generated user inputs (.json/.jsonl). If omitted, the script first tries PROJECT_ROOT/generated_user_inputs.json, then generates inputs on the fly.",
    )
    parser.add_argument(
        "--network-input",
        type=Path,
        default=None,
        help="Optional network-state JSON used when generating user inputs on the fly.",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Sample count when generating inputs on the fly.")
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=DEFAULT_TARGET_SUCCESS_RATE,
        help="Target average success rate when generating inputs on the fly.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed when generating inputs on the fly.")
    parser.add_argument("--context", type=str, default="", help="Default conversation context attached to every request.")
    parser.add_argument(
        "--allow-user-interaction",
        action="store_true",
        help="Allow the IEA to ask blocking clarification questions in the terminal for each request.",
    )
    parser.add_argument("--session-prefix", type=str, default="iea-sft", help="Prefix used to synthesize session_id.")
    parser.add_argument("--snapshot-id", type=str, default="iea-generated", help="Default snapshot_id for synthesized requests.")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507", help="IEA model name.")
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_output_path("intent_encoding", "supervised", "success", "iea_sft_v1.jsonl"),
        help="Output JSONL path for the built dataset.",
    )
    parser.add_argument(
        "--chatml-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "success", "iea_chatml_sft_v1.jsonl"),
        help="Output JSONL path for the ChatML success dataset.",
    )
    parser.add_argument(
        "--chatml-failure-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "failure", "iea_chatml_sft_v1.jsonl"),
        help="Output JSONL path for the ChatML failure dataset.",
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "supervised", "failure", "iea_sft_rejects_v1.jsonl"),
        help="Output JSONL path for rejected rows.",
    )
    parser.add_argument(
        "--chatml-reject-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "chatml", "failure", "iea_chatml_rejects_v1.jsonl"),
        help="Output JSONL path for ChatML builder rejects.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=dataset_dir("intent_encoding", "chatml") / "build_report_v1.json",
        help="Build report output path.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Optional staging root that will contain the temporary runtime/interfaces tree used by build_iea_sft_records.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    default_user_inputs_path = PROJECT_ROOT / "generated_user_inputs.json"
    user_inputs_path = args.user_inputs
    if user_inputs_path is None and default_user_inputs_path.exists():
        user_inputs_path = default_user_inputs_path

    records = load_user_input_records(
        user_inputs_path=user_inputs_path,
        network_input_path=args.network_input,
        count=args.count,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
    )
    agent = IntentEncodingAgent(model_name=args.model)

    chatml_records, chatml_rejects = build_chatml_records_from_user_inputs(
        agent=agent,
        records=records,
        session_prefix=args.session_prefix,
        snapshot_id=args.snapshot_id,
        default_context=args.context,
        project_root=PROJECT_ROOT,
        allow_user_interaction=args.allow_user_interaction,
    )
    chatml_success_records, chatml_failure_records = split_iea_chatml_records(chatml_records)
    write_jsonl(args.chatml_output, chatml_success_records)
    write_jsonl(args.chatml_failure_output, chatml_failure_records)
    write_jsonl(args.chatml_reject_output, chatml_rejects)
    merged_report = load_build_report(args.report_output)
    report = BuildReport(
        artifact_total=len(records),
        paired_total=len(chatml_records),
        iea_sft_samples=len(chatml_success_records),
        reject_total=len(chatml_failure_records) + _count_chatml_build_errors(chatml_rejects),
    )
    merged_report.artifact_total = report.artifact_total
    merged_report.paired_total = report.paired_total
    merged_report.iea_sft_samples = report.iea_sft_samples
    merged_report.reject_total = report.reject_total
    save_build_report(args.report_output, merged_report)

    print(f"IEA requests processed: {len(records)}")
    print(f"Wrote {len(chatml_success_records)} IEA ChatML success rows to {args.chatml_output}")
    print(f"Wrote {len(chatml_failure_records)} IEA ChatML failure rows to {args.chatml_failure_output}")
    print(f"ChatML rejects {len(chatml_rejects)} rows -> {args.chatml_reject_output}")
    print(f"Build report -> {args.report_output}")


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not 0.0 < args.target_success_rate < 1.0:
        raise ValueError("--target-success-rate must be between 0 and 1")
    run_pipeline(args)


if __name__ == "__main__":
    main()
