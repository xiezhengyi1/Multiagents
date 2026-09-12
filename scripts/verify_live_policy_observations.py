"""Read real live SM/AM/URSP through the typed Multiagents adapter without policy writes."""
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from control_runtime.integrations.testbed import TestbedClient, TestbedConfig, TestbedSimulatorAdapter
from control_runtime.simulation import ObservationRequest

run = Path(__file__).resolve().parents[2] / "oai-free5gc-testbed/artifacts/runs/s2-live-auto"
credentials = json.loads((run / "ue-policy/credentials.json").read_text())
scenario = yaml.safe_load((run / "source-scenario.yaml").read_text())
config = TestbedConfig(
    snapshot_base_url="http://127.0.0.1:8081", action_base_url="http://127.0.0.1:8080",
    subscriber_base_url="http://127.0.0.1:5000", run_id="s2-live-auto", plmn_id="20893",
    action_api_key=credentials["multiagents_key"], timeout_seconds=120,
)
with httpx.Client(timeout=120, trust_env=False) as http:
    adapter = TestbedSimulatorAdapter(client=TestbedClient(config=config, http=http))
    snapshot = adapter.capture_snapshot()
    observations = []
    for ue in scenario["ues"]:
        for query in ("get_sm_ue_flow_catalog", "get_am_policy_context", "inspect_mobility_ue_policies"):
            observation = adapter.observe(ObservationRequest(query_name=query, arguments={"supi": ue["supi"]}))
            result = observation.result
            if query == "get_sm_ue_flow_catalog":
                assert result["flow_catalog"], ue["supi"]
            else:
                assert result["association_id"], ue["supi"]
            if query == "inspect_mobility_ue_policies":
                assert len(result["policies"]) == 1
                assert result["policies"][0]["status"] == "complete"
            observations.append(observation.model_dump(mode="json"))
directory = run / "ue-policy/evidence" / datetime.now(timezone.utc).strftime("multiagents-live-%Y%m%dT%H%M%SZ")
directory.mkdir()
(directory / "observations.json").write_text(json.dumps(observations, indent=2))
print(json.dumps({"status": "verified", "ues": 6, "observations": len(observations), "evidence": str(directory)}))
