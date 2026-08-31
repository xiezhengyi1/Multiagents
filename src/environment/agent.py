from __future__ import annotations

import json
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel, Field

from shared.contracts import (
    AgentInvocation,
    AgentMessage,
    AgentRunResult,
    ExecutionScope,
    TraceMetadata,
    TraceTarget,
)
from experiments.scripts.common import resolve_python_executable

from .compiler import EnvironmentAgentCompiler, S3_QUALITY_PROFILE
from .contracts import EnvironmentGenerationRequest, ScenarioCandidate
from .prompts import ENVIRONMENT_DESIGN_AGENT_SYSTEM_PROMPT
from .tools import build_environment_tools
from shared.agents import BaseAgent, coerce_structured_response
from shared.logging import log_event, log_timing
from shared.runtime import (
    ArtifactEnvelope,
    ArtifactWorkerMixin,
    ContextPolicy,
    RuntimePersistence,
    ToolLoopExecutionError,
    extract_tool_results,
)
from application_runtime.persistence import create_runtime_persistence


ENVIRONMENT_TOOL_CAPABILITIES: Mapping[str, tuple[str, ...]] = MappingProxyType({})


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


def _extract_simulation_payloads(result: AgentRunResult) -> list[dict[str, Any]]:
    return _extract_tool_payloads(result, "simulate_candidate_environment")


def _extract_tool_payloads(result: AgentRunResult, tool_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for tool_result in extract_tool_results(result.messages):
        if str(tool_result.get("name") or "").strip() != tool_name:
            continue
        payload = json.loads(str(tool_result.get("content") or ""))
        if not isinstance(payload, dict):
            raise TypeError(f"tool {tool_name} must return a JSON object")
        payloads.append(payload)
    return payloads


def _load_mapping(path: Path) -> dict[str, Any]:
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping")
    return payload


def _load_written_candidate(payload: dict[str, Any]) -> ScenarioCandidate:
    scenario_path = Path(str(payload.get("scenario_path") or "")).resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"written scenario YAML does not exist: {scenario_path}")
    scenario = _load_mapping(scenario_path)
    overlay_path = Path(str(payload.get("split_mode_overlay_path") or "")).resolve()
    overlay = _load_mapping(overlay_path) if str(payload.get("split_mode_overlay_path") or "").strip() else None
    return ScenarioCandidate(
        scenario_id=str(payload.get("scenario_id") or scenario.get("scenario_id") or "").strip(),
        name=str(scenario.get("name") or "").strip(),
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
        model_name: str = "qwen3.6-flash",
        use_local_model: bool = False,
        *,
        initialize_runtime: bool = True,
        llm: Any = ...,
        environment_tools: list[Any] | None = None,
        scenario_root: Path | None = None,
        persistence: RuntimePersistence | None = None,
        quality_profile: str | None = S3_QUALITY_PROFILE,
    ) -> None:
        self.agent_name = "environment_generation"
        self.compiler = EnvironmentAgentCompiler()
        self.quality_profile = quality_profile
        self.compiler.quality_requirements(quality_profile)
        self.system_prompt = ENVIRONMENT_DESIGN_AGENT_SYSTEM_PROMPT
        self.environment_tools = environment_tools
        if not initialize_runtime:
            self.agent = None
            return
        super().__init__(model_name=model_name, use_local_model=use_local_model)
        if llm is not ...:
            self.llm = llm
        runtime_persistence = persistence if persistence is not None else create_runtime_persistence()
        self.initialize_agent_runtime(logger_color="\033[92m", persistence=runtime_persistence)
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
                quality_profile=self.quality_profile,
                semantic_design_mode=True,
                logger=self.logger,
            )
        self.agent = self.create_json_agent(
            tools=self.environment_tools,
            system_prompt=self.system_prompt,
            prompt_name="environment_generation_advisor",
            prompt_version="2.0.0",
            response_model=EnvironmentAdvisorOutput,
            max_iterations=14,
            context_policy=ContextPolicy(
                tool_result_char_limits={
                    "submit_environment_catalog": 5000,
                    "prepare_environment_design_context": 7000,
                    "inspect_environment_catalog": 5000,
                    "compile_environment_design": 6000,
                    "revise_environment_design": 6500,
                    "validate_environment_draft": 4000,
                }
            ),
            capability_aliases=ENVIRONMENT_TOOL_CAPABILITIES,
        )

    def expected_request_type(self) -> str:
        return "EnvironmentGenerationRequest"

    def response_artifact_type(self) -> str:
        return "ScenarioCandidate"

    def handle_artifact(self, envelope: ArtifactEnvelope) -> dict[str, Any]:
        request = EnvironmentGenerationRequest(**dict(envelope.payload or {}))
        candidate = self.generate_environment(
            request,
            session_id=envelope.scope.session_id,
            snapshot_id=envelope.scope.snapshot_id or "",
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
        prompt = self.compiler.build_semantic_generation_prompt(
            request,
            quality_profile=self.quality_profile,
        )
        prompt = (
            f"{prompt}\n\n"
            "Semantic closed-loop requirement:\n"
            "- Call prepare_environment_design_context once for examples and the trusted runtime shape.\n"
            "- Submit the complete semantic-key manifest with submit_environment_catalog.\n"
            "- Submit all topology decisions and exact flow metrics together with compile_environment_design.\n"
            "- The compiler derives identifiers and references only; it never adjusts your metrics.\n"
            "- If the first submission has a schema error, correct the complete semantic design once.\n"
            "- For completeness or quality errors, call revise_environment_design with only complete affected entities.\n"
            "- Use validate_environment_draft, write_validated_environment_yaml, and simulate_candidate_environment in that order.\n"
            "- Never call YAML validation, writing, or simulation before semantic compilation returns status=ok.\n"
            "- When simulation fails and returns failure_analysis, follow its suggested_section and guidance. Do NOT retry the same data unchanged.\n"
            "- If simulation fails but the draft appears correct in-memory, use read_back_written_yaml to compare the draft against the file on disk. Look for missing keys, type mismatches, or serialization gaps.\n"
            "- Compilation and simulation feedback is recorded automatically; do not spend a tool call restating it.\n"
            "- Simulation succeeds only when the real launcher starts, the live graph snapshot exists, the policy gateway is healthy, and SLA initialization passes.\n"
            f"- Enforce the {self.quality_profile or 'schema-only'} quality contract: "
            f"{self.compiler.quality_requirements(self.quality_profile)}\n"
            "- Do not emit final YAML or compiler-owned IDs; submit the semantic design through tools.\n"
            "- Final JSON must summarize the successfully validated environment candidate.\n"
            "- CRITICAL: your final JSON output MUST contain these top-level keys:\n"
            "    scenario_id (string), name (string), validation_status,\n"
            "    validation_feedback, tool_loop_summary, rationale.\n"
        )
        runtime_context = self.build_runtime_context(
            agent_name=self.agent_name,
            scope=ExecutionScope(
                session_id=session_id,
                thread_id=session_id,
                snapshot_id=snapshot_id or None,
            ),
        )
        invocation = AgentInvocation(
            messages=[AgentMessage(role="user", content=prompt)],
            metadata=TraceMetadata(
                path_label="environment_generation",
                prompt_name="environment_generation_advisor",
                prompt_version="2.0.0",
            ),
        )
        try:
            log_event(
                self.logger,
                "environment_advisor_invoke_start",
                model_name=self.model_name,
                max_iterations=14,
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            result = self.agent.invoke(invocation, context=runtime_context, defer_trace=True)
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
        try:
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
            candidate = _load_written_candidate(written_payloads[-1])
            report = self.compiler.validate_candidate(candidate, quality_profile=self.quality_profile)
            if not report.ok:
                raise RuntimeError(
                    "Environment advisor returned invalid scenario: "
                    + json.dumps(report.errors, ensure_ascii=False)
                )
        except Exception as exc:
            self.agent.write_trace(
                invocation=invocation,
                context=runtime_context,
                result=result,
                error=str(exc),
            )
            log_timing(
                self.logger,
                "environment_total",
                time.perf_counter() - total_start,
                status="error",
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            raise
        self.agent.write_trace(
            invocation=invocation,
            context=runtime_context,
            result=result,
            derived_targets=[
                TraceTarget(
                    kind="validated_output",
                    producer="validator",
                    schema_name=ScenarioCandidate.__name__,
                    output_schema=ScenarioCandidate.model_json_schema(),
                    value=candidate.model_dump(mode="json"),
                )
            ],
        )
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
