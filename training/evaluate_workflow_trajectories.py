from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from shared.agents import BaseAgent
from training.artifacts import PROJECT_ROOT, processed_dir, read_jsonl, write_json, write_jsonl
from training.projection import load_projected_trace_records
from training.schemas import ProjectedTraceRecord, WorkflowTrajectoryRecord


class ToolIssue(BaseModel):
    severity: str
    category: str
    detail: str


class ToolCorrectnessJudgement(BaseModel):
    applicable: bool = True
    passed: bool = True
    summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    issues: List[ToolIssue] = Field(default_factory=list)


class WorkflowTrajectoryJudgement(BaseModel):
    passed: bool = True
    summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


TOOL_CORRECTNESS_PROMPT = """
You are a strict evaluator for one agent trajectory inside a larger workflow.

Evaluate only tool-usage correctness for this single trajectory.

A trajectory passes only when all of the following are true:
1. The tool calls are necessary for the information gap in the trace.
2. The chosen tools are appropriate for that gap.
3. Tool arguments are grounded in prior messages or tool outputs.
4. The trajectory does not ignore a tool result in a way that makes the next step unjustified.
5. The trajectory does not contain clearly redundant repeated business-tool calls.

Rules:
- Ignore purely internal reasoning tools such as think or think_tool.
- Do not judge business quality beyond what is needed to judge tool choice.
- If a trace has no business tool calls, set applicable=false and do not invent failures.
- Be concrete. Every issue must point to an actual mistake visible in the provided payload.

Return only the requested structured JSON.
""".strip()


WORKFLOW_TRAJECTORY_PROMPT = """
You are a strict evaluator for an end-to-end multi-agent workflow trajectory.

You will receive:
- the original user input,
- the workflow run result,
- per-agent trace summaries,
- per-agent tool-correctness judgements.

Evaluate whether the overall trajectory is coherent and operationally sound.

A workflow passes only when:
1. The workflow progressed through the needed agents without unexplained gaps.
2. The final outcome is consistent with the evidence gathered during the trajectory.
3. Any failed or incomplete step is explicitly reflected in the final workflow result.
4. There is no obvious contradiction between agent outputs across the same session.

Rules:
- Focus on the trajectory, not on polishing or stylistic issues.
- If the workflow run itself failed, do not automatically fail the judgement; fail only when the trajectory handling is unsound.
- Keep issues concrete and tied to the payload.

Return only the requested structured JSON.
""".strip()


def discover_trace_files(root: Path) -> Dict[str, Path]:
    trace_files: Dict[str, Path] = {}
    training_root = root / "training"
    if not training_root.exists():
        return trace_files
    for child in training_root.iterdir():
        if not child.is_dir():
            continue
        agent_name = child.name
        trace_path = child / "raw_traces" / f"{agent_name}.jsonl"
        if trace_path.exists():
            trace_files[agent_name] = trace_path
    return trace_files


def load_traces_by_session(root: Path) -> Dict[str, Dict[str, List[ProjectedTraceRecord]]]:
    by_session: Dict[str, Dict[str, List[ProjectedTraceRecord]]] = {}
    for agent_name, trace_path in discover_trace_files(root).items():
        for trace in load_projected_trace_records(trace_path):
            session_id = trace.scope.session_id
            by_session.setdefault(session_id, {}).setdefault(agent_name, []).append(trace)
    return by_session


def _business_tool_calls(trace: ProjectedTraceRecord) -> List[Dict[str, Any]]:
    return [
        dict(call)
        for call in trace.tool_calls
        if str(call.get("name") or "").strip() not in {"think", "think_tool"}
    ]


def _structural_issues(trace: ProjectedTraceRecord) -> List[ToolIssue]:
    issues: List[ToolIssue] = []
    tool_calls = _business_tool_calls(trace)
    tool_results_by_id = {
        str(item.get("tool_call_id") or "").strip(): item
        for item in trace.tool_results
        if str(item.get("tool_call_id") or "").strip()
    }

    seen_signatures: set[str] = set()
    for call in tool_calls:
        call_id = str(call.get("id") or "").strip()
        tool_name = str(call.get("name") or "").strip()
        args = call.get("args") or {}
        signature = f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if signature in seen_signatures:
            issues.append(
                ToolIssue(
                    severity="warning",
                    category="duplicate_call",
                    detail=f"duplicate business tool call detected: {tool_name}",
                )
            )
        else:
            seen_signatures.add(signature)
        if not call_id:
            issues.append(
                ToolIssue(
                    severity="error",
                    category="missing_call_id",
                    detail=f"tool call for {tool_name} is missing an id",
                )
            )
            continue
        if call_id not in tool_results_by_id and trace.status == "success":
            issues.append(
                ToolIssue(
                    severity="error",
                    category="missing_tool_result",
                    detail=f"successful trace is missing tool result for {tool_name} ({call_id})",
                )
            )

    return issues


def _compact_trace_payload(trace: ProjectedTraceRecord) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "agent_name": trace.agent_name,
        "status": trace.status,
        "messages": [message.model_dump(mode="json") for message in trace.messages],
        "tool_calls": _business_tool_calls(trace),
        "tool_results": trace.tool_results,
        "structured_response": trace.structured_response,
        "error": trace.error,
    }


def evaluate_trace_tool_correctness(trace: ProjectedTraceRecord, llm: Any) -> Dict[str, Any]:
    structural_issues = _structural_issues(trace)
    business_tools = _business_tool_calls(trace)
    if not business_tools:
        return {
            "trace_id": trace.trace_id,
            "agent_name": trace.agent_name,
            "applicable": False,
            "structural_issues": [item.model_dump(mode="json") for item in structural_issues],
            "llm_judgement": ToolCorrectnessJudgement(applicable=False, passed=True, summary="no business tool calls in this trajectory").model_dump(mode="json"),
        }

    runner = llm.with_structured_output(ToolCorrectnessJudgement)
    judgement = runner.invoke(
        [
            {"role": "system", "content": TOOL_CORRECTNESS_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "trace": _compact_trace_payload(trace),
                        "structural_issues": [item.model_dump(mode="json") for item in structural_issues],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
    )
    normalized = judgement if isinstance(judgement, ToolCorrectnessJudgement) else ToolCorrectnessJudgement.model_validate(judgement)
    if any(item.severity == "error" for item in structural_issues):
        normalized.passed = False
    return {
        "trace_id": trace.trace_id,
        "agent_name": trace.agent_name,
        "applicable": True,
        "structural_issues": [item.model_dump(mode="json") for item in structural_issues],
        "llm_judgement": normalized.model_dump(mode="json"),
    }


def _trace_summary(trace: ProjectedTraceRecord) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "agent_name": trace.agent_name,
        "status": trace.status,
        "tool_names": [str(call.get("name") or "") for call in _business_tool_calls(trace)],
        "structured_response": trace.structured_response,
        "error": trace.error,
    }


def evaluate_workflow_record(
    record: WorkflowTrajectoryRecord,
    session_traces: Dict[str, List[ProjectedTraceRecord]],
    llm: Any,
) -> Dict[str, Any]:
    agent_evaluations: List[Dict[str, Any]] = []
    for agent_name in sorted(session_traces.keys()):
        traces = session_traces[agent_name]
        for trace in traces:
            agent_evaluations.append(
                {
                    "agent_name": agent_name,
                    "trace_summary": _trace_summary(trace),
                    "tool_correctness": evaluate_trace_tool_correctness(trace, llm),
                }
            )

    runner = llm.with_structured_output(WorkflowTrajectoryJudgement)
    workflow_judgement = runner.invoke(
        [
            {"role": "system", "content": WORKFLOW_TRAJECTORY_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "workflow_record": record.model_dump(mode="json"),
                        "agent_evaluations": agent_evaluations,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
    )
    normalized = workflow_judgement if isinstance(workflow_judgement, WorkflowTrajectoryJudgement) else WorkflowTrajectoryJudgement.model_validate(workflow_judgement)
    return {
        "record_index": record.record_index,
        "scenario_id": record.scope.scenario_id,
        "scenario_tags": record.scope.scenario_tags,
        "session_id": record.scope.session_id,
        "status": record.status,
        "completed": record.completed,
        "user_input": record.user_input,
        "agent_evaluations": agent_evaluations,
        "workflow_judgement": normalized.model_dump(mode="json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate collected workflow trajectories for tool correctness and end-to-end trajectory quality.",
    )
    parser.add_argument(
        "--run-records",
        type=Path,
        default=processed_dir("workflow") / "trajectory_runs_v2.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir("workflow") / "trajectory_evaluations_v2.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=processed_dir("workflow") / "trajectory_evaluation_summary_v2.json",
    )
    parser.add_argument("--model", type=str, default="qwen-plus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_records = [
        WorkflowTrajectoryRecord.model_validate(row)
        for row in read_jsonl(args.run_records)
    ]
    session_traces = load_traces_by_session(PROJECT_ROOT)
    llm = BaseAgent(model_name=args.model, temperature=0.0).get_llm()

    evaluation_rows: List[Dict[str, Any]] = []
    for record in run_records:
        session_id = record.scope.session_id
        evaluation_rows.append(
            evaluate_workflow_record(record, session_traces.get(session_id, {}), llm)
        )

    write_jsonl(args.output, evaluation_rows)

    total = len(evaluation_rows)
    workflow_passed = sum(1 for row in evaluation_rows if (row.get("workflow_judgement") or {}).get("passed"))
    tool_eval_total = 0
    tool_eval_passed = 0
    for row in evaluation_rows:
        for agent_eval in row.get("agent_evaluations", []):
            tool_correctness = agent_eval.get("tool_correctness") or {}
            if not tool_correctness.get("applicable"):
                continue
            tool_eval_total += 1
            if (tool_correctness.get("llm_judgement") or {}).get("passed"):
                tool_eval_passed += 1

    summary = {
        "total_workflow_records": total,
        "workflow_passed": workflow_passed,
        "workflow_pass_rate": round(workflow_passed / total, 4) if total else 0.0,
        "tool_eval_total": tool_eval_total,
        "tool_eval_passed": tool_eval_passed,
        "tool_eval_pass_rate": round(tool_eval_passed / tool_eval_total, 4) if tool_eval_total else 0.0,
        "output": str(args.output),
    }
    write_json(args.summary_output, summary)

    print(f"Evaluated {total} workflow records")
    print(f"Workflow pass rate: {summary['workflow_pass_rate']:.2%}")
    print(f"Tool-eval pass rate: {summary['tool_eval_pass_rate']:.2%}")
    print(f"Detailed results -> {args.output}")
    print(f"Summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
