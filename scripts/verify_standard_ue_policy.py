"""Run typed Multiagents AM/URSP actions against the explicit isolated testbed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from control_runtime.integrations.testbed import (
    TestbedClient,
    TestbedConfig,
    TestbedSimulatorAdapter,
)
from control_runtime.simulation import ExecutionAction, ExecutionOperation
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.UrspRuleRequest import UrspRuleRequest

run = (
    Path(__file__).resolve().parents[2]
    / "oai-free5gc-testbed/artifacts/runs/phase2-isolated"
)
credentials = json.loads((run / "ue-policy/credentials.json").read_text())
scenario = yaml.safe_load((run / "source-scenario.yaml").read_text())
config = TestbedConfig(
    snapshot_base_url="http://127.0.0.1:18081",
    action_base_url="http://127.0.0.1:18080",
    subscriber_base_url="http://127.0.0.1:15000",
    run_id="phase2-isolated",
    plmn_id="20893",
    action_api_key=credentials["multiagents_key"],
    timeout_seconds=120,
)
client = TestbedClient(config=config, http=httpx.Client(timeout=120, trust_env=False))
adapter = TestbedSimulatorAdapter(client=client)
adapter.capture_snapshot()
native = client.latest_snapshot()
flow = next(row for row in native["flows"] if row["traffic"]["five_tuple"][4] == "tcp")
ue = scenario["ues"][0]
target = next(row for row in ue["sessions"] if row["dnn"] != flow["session"]["dnn"])
slice_ = next(row for row in scenario["slices"] if row["id"] == target["slice_id"])
am = native["ue_policies"][0]["am_policy_associations"][0]
operations = [
    ExecutionOperation(
        operation_id="standard-ursp",
        domain="mobility",
        operation="update",
        resource_type="UrspRuleRequest",
        resource_id=f"ursp-{flow['app_id']}-{flow['id']}",
        payload=UrspRuleRequest.model_validate(
            {
                "relatPrecedence": flow["ue_policy_route"]["rule_precedence"],
                "trafficDesc": {
                    "flowDescs": [
                        f"permit out tcp from any to {flow['traffic']['five_tuple'][1]} 8080"
                    ]
                },
                "routeSelParamSets": [
                    {
                        "precedence": 1,
                        "dnn": target["dnn"],
                        "snssai": {"sst": slice_["sst"], "sd": slice_["sd"]},
                    }
                ],
            }
        ),
    ),
    ExecutionOperation(
        operation_id="standard-am",
        domain="mobility",
        operation="update",
        resource_type="PcfAmPolicyControlPolicyAssociation",
        resource_id=am["id"],
        payload=PcfAmPolicyControlPolicyAssociation.model_validate(
            {
                "request": {
                    "supi": ue["supi"],
                    "notificationUri": "http://unused.invalid",
                    "suppFeat": "0",
                },
                "rfsp": 65 if am["rfsp"] != 65 else 64,
                "suppFeat": "0",
            }
        ),
    ),
]
action = ExecutionAction(operations=operations)
feedback = adapter.execute(action)
directory = (
    run
    / "ue-policy/evidence"
    / datetime.now(timezone.utc).strftime("multiagents-%Y%m%dT%H%M%S.%fZ")
)
directory.mkdir()
(directory / "action.json").write_text(action.model_dump_json(indent=2))
(directory / "feedback.json").write_text(feedback.model_dump_json(indent=2))
assert feedback.status == "success", feedback.receipts
print(
    json.dumps(
        {
            "status": feedback.status,
            "operations": len(feedback.receipts),
            "evidence": str(directory),
        }
    )
)
