from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from experiments.scripts.common import resolve_python_executable

from .compiler import EnvironmentAgentCompiler
from .contracts import EnvironmentGenerationRequest, ScenarioCandidate
from .prompts import ENVIRONMENT_AGENT_SYSTEM_PROMPT
from .tools import build_environment_tools
from shared.agents import BaseAgent, coerce_structured_response
from shared.logging import log_event, log_timing
from shared.runtime import ArtifactEnvelope, ArtifactWorkerMixin, ToolLoopExecutionError, extract_tool_results


def simulation_validation_succeeded(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        if (
            payload.get("status") == "ok"
            and payload.get("simulator_started") is True
            and (payload.get("graph_snapshot") or {}).get("ok") is True
            and (payload.get("gateway_health") or {}).get("ok") is True
            and (payload.get("sla_initialization") or {}).get("ok") is True
        ):
            return True
    return False


def _resolve_environment_python_executables(project_root: Path, workspace_root: Path) -> tuple[Path, Path]:
    return (
        resolve_python_executable(project_root),
        resolve_python_executable(workspace_root / "ns3-free5gc-integration"),
    )


def _extract_simulation_payloads(result: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_tool_payloads(result, "simulate_candidate_environment")


def _extract_tool_payloads(result: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for tool_result in extract_tool_results(result.get("messages") or []):
        if str(tool_result.get("name") or "").strip() != tool_name:
            continue
        try:
            payload = json.loads(str(tool_result.get("content") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except Exception:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping")
    return payload


def _load_written_candidate(payload: dict[str, Any], *, fallback_name: str = "") -> ScenarioCandidate:
    scenario_path = Path(str(payload.get("scenario_path") or "")).resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"written scenario YAML does not exist: {scenario_path}")
    scenario = _load_mapping(scenario_path)
    overlay_path = Path(str(payload.get("split_mode_overlay_path") or "")).resolve()
    overlay = _load_mapping(overlay_path) if str(payload.get("split_mode_overlay_path") or "").strip() else None
    return ScenarioCandidate(
        scenario_id=str(payload.get("scenario_id") or scenario.get("scenario_id") or "").strip(),
        name=str(scenario.get("name") or fallback_name or "").strip(),
        scenario=scenario,
        split_mode_overlay=overlay,
        source_path=scenario_path,
    )


class EnvironmentAdvisorOutput(BaseModel):
    scenario_id: str = Field(description="Generated scenario id")
    name: str = Field(description="Human readable scenario name")
    validation_status: str = Field(default="unknown")
    validation_feedback: list[str] = Field(default_factory=list)
    tool_loop_summary: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")


class EnvironmentGenerationAgent(BaseAgent, ArtifactWorkerMixin):
    agent_name = "environment_generation"

    def __init__(
        self,
        model_name: str = "qwen3-30b-a3b-instruct-2507",
        use_local_model: bool = False,
        *,
        initialize_runtime: bool = True,
        llm: Any = ...,
        environment_tools: list[Any] | None = None,
        scenario_root: Path | None = None,
    ) -> None:
        self.agent_name = "environment_generation"
        self.compiler = EnvironmentAgentCompiler()
        self.system_prompt = ENVIRONMENT_AGENT_SYSTEM_PROMPT
        self.environment_tools = environment_tools
        if not initialize_runtime:
            self.agent = None
            return
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        if llm is not ...:
            self.llm = llm
        self.initialize_agent_runtime(logger_color="\033[92m")
        if self.environment_tools is None:
            project_root = Path.cwd()
            workspace_root = project_root.parent
            python_executable, stack_python = _resolve_environment_python_executables(project_root, workspace_root)
            from .launcher import EnvironmentLauncher

            self.environment_tools = build_environment_tools(
                compiler=self.compiler,
                launcher=EnvironmentLauncher(
                    project_root=project_root,
                    workspace_root=workspace_root,
                    python_executable=python_executable,
                    stack_python_executable=stack_python,
                ),
                scenario_root=scenario_root or (project_root / "experiments" / "scenarios"),
                execute_simulator=True,
            )
        self.agent = self.create_json_agent(
            tools=self.environment_tools,
            system_prompt=self.system_prompt,
            response_model=EnvironmentAdvisorOutput,
            max_iterations=32,
            tool_error_mode="return",
            max_calls_per_tool=5,
            tool_call_limits={
                "replace_draft_section": 12,
                "patch_draft_entity": 12,
                "inspect_draft_section": 8,
            },
        )

    def expected_request_type(self) -> str:
        return "EnvironmentGenerationRequest"

    def response_artifact_type(self) -> str:
        return "ScenarioCandidate"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> dict[str, Any]:
        request = EnvironmentGenerationRequest(**dict(envelope.payload or {}))
        candidate = self.generate_environment(
            request,
            session_id=envelope.session_id,
            snapshot_id=envelope.snapshot_id,
        )
        return {
            "scenario_id": candidate.scenario_id,
            "name": candidate.name,
            "scenario": candidate.scenario,
            "split_mode_overlay": candidate.split_mode_overlay,
        }

    def generate_environment(
        self,
        request: EnvironmentGenerationRequest,
        *,
        session_id: str = "",
        snapshot_id: str = "",
    ) -> ScenarioCandidate:
        self.ensure_worker_runtime_initialized()
        total_start = time.perf_counter()
        log_event(
            self.logger,
            "environment_generate_start",
            session_id=session_id,
            snapshot_id=snapshot_id,
        )
        prompt = self.compiler.build_generation_prompt(request)
        prompt = (
            f"{prompt}\n\n"
            "Closed-loop requirement:\n"
            "- You must call list_existing_environment_specs before generating the candidate.\n"
            "- Initialize metadata with initialize_environment_draft.\n"
            "- Populate ordered sections with replace_draft_section; use patch_draft_entity and inspect_draft_section only for focused repairs.\n"
            "- Use validate_environment_draft, write_validated_environment_yaml, and simulate_candidate_environment in that order.\n"
            "- If validation or simulation reports a failure, call record_validation_feedback and produce a revised candidate.\n"
            "- Simulation succeeds only when the real launcher starts, the live graph snapshot exists, the policy gateway is healthy, and SLA initialization passes.\n"
            "- Do not submit a complete scenario mapping during draft mutation or in the final JSON.\n"
            "- Final JSON must summarize the successfully validated environment candidate.\n"
            "- CRITICAL: your final JSON output MUST contain these top-level keys:\n"
            "    scenario_id (string), name (string), validation_status,\n"
            "    validation_feedback, tool_loop_summary, rationale.\n"
        )
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            session_id=session_id,
            snapshot_id=snapshot_id,
            thread_id=session_id,
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "trace_write_mode": "manual",
            "trace_metadata": {"path_label": "environment_generation"},
        }
        try:
            result = self.agent.invoke(payload, context=runtime_context)
        except Exception as exc:
            self.logger.exception(f"Failed to generate environment candidate: {exc}")
            log_timing(
                self.logger,
                "environment_total",
                time.perf_counter() - total_start,
                status="error",
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            if isinstance(exc, ToolLoopExecutionError):
                raise RuntimeError(f"Environment generation advisor failed: {exc}") from exc
            raise
        simulation_payloads = _extract_simulation_payloads(result)
        if not simulation_validation_succeeded(simulation_payloads):
            raise RuntimeError("Environment advisor returned without successful real simulator validation evidence")
        output = coerce_structured_response(
            result,
            EnvironmentAdvisorOutput,
            error_message="Environment advisor returned no structured_response",
        )
        written_payloads = _extract_tool_payloads(result, "write_validated_environment_yaml")
        if not written_payloads:
            raise RuntimeError("Environment advisor returned without validated YAML write evidence")
        candidate = _load_written_candidate(written_payloads[-1], fallback_name=output.name)
        report = self.compiler.validate_candidate(candidate)
        if not report.ok:
            log_timing(
                self.logger,
                "environment_total",
                time.perf_counter() - total_start,
                status="error",
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            raise RuntimeError("Environment advisor returned invalid scenario: " + json.dumps(report.errors, ensure_ascii=False))
        log_timing(
            self.logger,
            "environment_total",
            time.perf_counter() - total_start,
            status="success",
            session_id=session_id,
            snapshot_id=snapshot_id,
            scenario_id=candidate.scenario_id,
        )
        return candidate


__all__ = ["EnvironmentAdvisorOutput", "EnvironmentGenerationAgent", "simulation_validation_succeeded"]
