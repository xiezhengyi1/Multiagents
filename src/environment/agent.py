from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .compiler import EnvironmentAgentCompiler
from .contracts import EnvironmentGenerationRequest, ScenarioCandidate
from .prompts import ENVIRONMENT_AGENT_SYSTEM_PROMPT
from .tools import build_environment_tools

from shared.agents import BaseAgent, coerce_structured_response
from shared.logging import log_event, log_timing
from shared.runtime import ArtifactEnvelope, ArtifactWorkerMixin, ToolLoopExecutionError


class EnvironmentAdvisorOutput(BaseModel):
    scenario_id: str = Field(description="Generated scenario id")
    name: str = Field(description="Human readable scenario name")
    scenario: dict[str, Any] = Field(description="Complete base scenario YAML payload")
    split_mode_overlay: dict[str, Any] | None = Field(default=None)
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
            python_executable = project_root / ".venv" / "Scripts" / "python.exe"
            stack_python = workspace_root / "ns3-free5gc-integration" / ".venv" / "Scripts" / "python.exe"
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
                execute_simulator=False,
            )
        self.agent = self.create_json_agent(
            tools=self.environment_tools,
            system_prompt=self.system_prompt,
            response_model=EnvironmentAdvisorOutput,
            max_iterations=18,
            tool_error_mode="return",
            max_calls_per_tool=5,
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
            "- Use write_candidate_environment_yaml, validate_candidate_environment, and simulate_candidate_environment.\n"
            "- If validation or simulation reports a failure, call record_validation_feedback and produce a revised candidate.\n"
            "- Final JSON must describe the successfully validated environment candidate.\n"
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
        output = coerce_structured_response(
            result,
            EnvironmentAdvisorOutput,
            error_message="Environment advisor returned no structured_response",
        )
        candidate = ScenarioCandidate(
            scenario_id=output.scenario_id,
            name=output.name,
            scenario=output.scenario,
            split_mode_overlay=output.split_mode_overlay,
        )
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


__all__ = ["EnvironmentAdvisorOutput", "EnvironmentGenerationAgent"]
