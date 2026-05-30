from environment import EnvironmentGenerationAgent, EnvironmentGenerationRequest
from pathlib import Path

agent = EnvironmentGenerationAgent(
    model_name="qwen3-30b-a3b-instruct-2507",
    scenario_root=Path("experiments/scenarios"),
)

request = EnvironmentGenerationRequest(
    scenario_id="G001",
    objective="Generate a URLLC stress scenario with slice contention",
    complexity="medium",
    target_flow_count=8,
    topology_mode="ulcl",
    stress_mode="slice_resource_contention",
)

candidate = agent.generate_environment(request)