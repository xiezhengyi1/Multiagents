from environment import EnvironmentGenerationAgent, EnvironmentGenerationRequest
from pathlib import Path

agent = EnvironmentGenerationAgent(
    model_name="qwen3-30b-a3b-instruct-2507",
    scenario_root=Path("experiments/scenarios"),
)

request = EnvironmentGenerationRequest(
    scenario_id="G001-S3",
    objective="Generate an S3-class multi-slice ULCL stress scenario with resilient QoS contention",
    complexity="high",
    target_flow_count=15,
    topology_mode="ulcl",
    stress_mode="slice_resource_contention",
    output_dir=Path("generated"),
)

candidate = agent.generate_environment(request)
