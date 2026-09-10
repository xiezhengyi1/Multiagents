from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer

from control_runtime.actors import OutcomeEvaluation
from control_runtime.graph import RepairLimits
from control_runtime.integrations import create_live_testbed_adapter
from control_runtime.simulation import ExecutionFeedback
from control_runtime.workflow import (
    ControlModels,
    ControlTools,
    ControlWorkflow,
    ModelBinding,
)
from experiments.contracts import ExperimentCase
from experiments.runner import PreparedEpisode


class LiveOutcomeEvaluator:
    def evaluate(self, feedback: ExecutionFeedback) -> OutcomeEvaluation:
        achieved = feedback.status == "success"
        return OutcomeEvaluation(
            achieved=achieved,
            reason=f"live testbed execution status: {feedback.status}",
            reward_components={"live_execution": 1.0 if achieved else -1.0},
        )


@dataclass(frozen=True, slots=True)
class LiveTestbedBackend:
    dataset_root: Path
    model: BaseChatModel
    model_name: str
    count_tokens: Callable[[str], int]

    def prepare(self, case: ExperimentCase) -> PreparedEpisode:
        if case.method != "multi_agent_control":
            raise ValueError("live testbed backend supports only multi_agent_control")
        workflow = ControlWorkflow.create(
            dataset_root=self.dataset_root,
            episode_id=case.episode_id,
            simulator_adapter=create_live_testbed_adapter(),
            models=ControlModels(
                main=ModelBinding(model=self.model, name=self.model_name),
                sba=ModelBinding(model=self.model, name=self.model_name),
                ppa=ModelBinding(model=self.model, name=self.model_name),
            ),
            tools=ControlTools(main=(), sba=(), ppa=()),
            repair_limits=RepairLimits(
                local_limits={"main": 2, "sba": 2, "ppa": 2},
                upstream={
                    "main": "episode_controller",
                    "sba": "main",
                    "ppa": "sba",
                },
                total_limit=6,
                controller_actor="episode_controller",
            ),
            outcome_evaluator=LiveOutcomeEvaluator(),
            budget_exhausted_reward_components={"repair_budget": -1.0},
            context_token_budget=6_000,
            count_tokens=self.count_tokens,
            transport_max_attempts=2,
        )
        user_input = cast(str, case.user_input)
        return PreparedEpisode(
            episode_directory=workflow.journal.directory,
            execute=lambda: workflow.run(user_input=user_input),
        )


def create_backend() -> LiveTestbedBackend:
    model_name = os.environ["vLLM_MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["VLLM_MODEL_PATH"],
        local_files_only=True,
    )
    model = ChatOpenAI(
        model=model_name,
        base_url=os.environ["vLLM_URL"],
        api_key=os.environ["VLLM_API_KEY"],
        timeout=float(os.environ["VLLM_TIMEOUT_SECONDS"]),
        temperature=0,
        max_tokens=1024,
        max_retries=0,
        model_kwargs={"parallel_tool_calls": False},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return LiveTestbedBackend(
        dataset_root=Path(os.environ["MULTIAGENTS_EPISODE_ROOT"]),
        model=model,
        model_name=model_name,
        count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )


__all__ = ["LiveOutcomeEvaluator", "LiveTestbedBackend", "create_backend"]
