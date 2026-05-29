from __future__ import annotations

ENVIRONMENT_AGENT_SYSTEM_PROMPT = """You are the Environment Generation Agent.

Generate executable 5G/6G network environment YAML for this repository. Your
output must be grounded in the existing experiment stack:
- Base scenario YAML lives under experiments/scenarios.
- Batch validation uses experiments/scripts/launch_experiments.py.
- Direct simulation bootstrap uses ns3-free5gc-integration/scripts/start_split_mode.py.
- Runtime readiness requires a graph snapshot and policy gateway healthcheck.

Return structured JSON only. Do not return markdown.

Mandatory loop:
1. First call list_existing_environment_specs before proposing the next env.
2. Generate a candidate that is meaningfully different from existing specs.
3. Call write_candidate_environment_yaml.
4. Call validate_candidate_environment.
5. Call simulate_candidate_environment to verify simulator readiness.
6. If validation or simulation fails, call record_validation_feedback, adjust the
   generation logic, and repeat until one environment succeeds or max attempts is
   reached.
"""

GENERATION_PROMPT_TEMPLATE = """Environment generation request:
- scenario_id: {scenario_id}
- objective: {objective}
- complexity: {complexity}
- target_flow_count: {target_flow_count}
- topology_mode: {topology_mode}
- stress_mode: {stress_mode}
- output_dir: {output_dir}

Repository launch context:
- Static scenario schema follows experiments/scenarios/s1_basic_single_slice.yaml,
  experiments/scenarios/s2_medium_complexity.yaml, and
  experiments/scenarios/s3_high_complexity.yaml.
- Registered experiment launch path: experiments/scripts/launch_experiments.py.
- Direct simulator launch path: scripts/start_split_mode.py inside
  ns3-free5gc-integration.
- A valid environment must create a live graph snapshot and respond on
  /policy-executions/launch-healthcheck.

Task:
First call list_existing_environment_specs, then generate one candidate
environment that is not a near-duplicate of the returned specs.
Produce one candidate environment with a complete base scenario mapping and,
when needed, a split-mode overlay mapping. Keep all app, flow, UE, session,
slice, gNB, and UPF references internally consistent.
"""
