from __future__ import annotations

import pytest
from pydantic import ValidationError

from control_runtime.actors.ppa import PpaProposal
from control_runtime.actors.sba import SbaArtifact
from control_runtime.execution import compile_ppa_execution_action
from control_runtime.simulation import ExecutionAction
from model import (
    PcfAmPolicyControlPolicyAssociation,
    SmPolicyDecision,
    UrspRuleRequest,
)


def test_ppa_compiler_emits_typed_3gpp_policy_contracts() -> None:
    sba_artifact = SbaArtifact.model_validate(
        {
            "supi": "imsi-001010000000001",
            "flows": [
                {
                    "app_id": "app-video",
                    "flow_id": "flow-1",
                    "name": "video",
                    "resolution_status": "resolved",
                }
            ],
            "mobility_intent": {
                "association_id": "am-1",
                "rfsp": 10,
                "allowed_snssais": ["01a1b2c3"],
                "target_snssais": ["01a1b2c3"],
                "access_type": "3GPP_ACCESS",
                "service_area_restriction": None,
            },
            "qos_operation_constraints": [
                {
                    "flow_id": "flow-1",
                    "app_id": "app-video",
                    "operation_type": "qos_reallocation",
                    "require_slice_change": False,
                    "target_slice_preference": "01a1b2c3",
                    "no_op_allowed": False,
                    "rationale": ["latency objective"],
                }
            ],
            "slice_migration_authorization": {
                "decision": "not_applicable",
                "target_snssais": [],
                "subscription_change_required": False,
                "rationale": ["QoS-only request"],
            },
            "open_questions": [],
        }
    )
    proposal = PpaProposal.model_validate(
        {
            "planning_status": "executable_plan",
            "rationale": "Compile every selected policy domain",
            "missing_evidence": [],
            "blocked_targets": [],
            "upstream_requests": [],
            "planner_conflicts": [],
            "open_questions": [],
            "sm_policies": [
                {
                    "flow_id": "flow-1",
                    "app_id": "app-video",
                    "priority": 5,
                    "target_latency_ms": 20,
                    "packet_error_rate": 0.001,
                    "max_br_ul_mbps": 50,
                    "max_br_dl_mbps": 100,
                    "gbr_ul_mbps": 10,
                    "gbr_dl_mbps": 20,
                    "target_jitter_ms": 5,
                    "flow_description": "permit out ip from any to assigned",
                }
            ],
            "am_policy": {
                "triggers": ["RFSP_CH"],
                "rfsp": 10,
                "allowed_snssais": [{"sst": 1, "sd": "a1b2c3"}],
                "target_snssais": [{"sst": 1, "sd": "a1b2c3"}],
                "ue_ambr_ul_mbps": 100,
                "ue_ambr_dl_mbps": 200,
                "notification_uri": "https://pcf.example/am",
                "supp_feat": "1",
            },
            "ursp_policies": [
                {
                    "target_type": "flow",
                    "app_id": "app-video",
                    "flow_id": "flow-1",
                    "relat_precedence": 1,
                    "traffic_desc": {
                        "flow_descs": ["permit out ip from any to assigned"],
                        "dnns": ["internet"],
                        "app_ids": ["video"],
                        "os_id": "android",
                    },
                    "route_sel_param_sets": [
                        {
                            "dnn": "internet",
                            "precedence": 1,
                            "snssai": {"sst": 1, "sd": "a1b2c3"},
                        }
                    ],
                }
            ],
            "partial_policies": [],
        }
    )

    action = compile_ppa_execution_action(
        proposal=proposal,
        sba_artifact=sba_artifact,
    )

    assert [type(operation.payload) for operation in action.operations] == [
        PcfAmPolicyControlPolicyAssociation,
        SmPolicyDecision,
        UrspRuleRequest,
    ]
    serialized = action.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert set(serialized["operations"][1]["payload"]) == {"pccRules", "qosDecs"}
    assert set(serialized["operations"][2]["payload"]) == {
        "trafficDesc",
        "relatPrecedence",
        "routeSelParamSets",
    }

    restored = ExecutionAction.model_validate_json(
        action.model_dump_json(by_alias=True, exclude_none=True)
    )
    assert [type(operation.payload) for operation in restored.operations] == [
        PcfAmPolicyControlPolicyAssociation,
        SmPolicyDecision,
        UrspRuleRequest,
    ]


def test_3gpp_output_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SmPolicyDecision.model_validate({"unknown": "ignored-before-refactor"})
    with pytest.raises(ValidationError):
        UrspRuleRequest.model_validate(
            {
                "relatPrecedence": 1,
                "routeSelParamSets": [
                    {"dnn": "internet", "precedence": 1, "unknown": True}
                ],
            }
        )
