from types import SimpleNamespace

import pytest

from control_runtime.integrations.testbed.ue_policy import project_policy_action
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.UrspRuleRequest import UrspRuleRequest


def context():
    return {
        "snapshot_id": "s1",
        "flows": [
            {
                "id": "f1",
                "app_id": "app",
                "supi": "imsi-208930000000009",
                "ue_policy_route": {"rule_precedence": 1},
            }
        ],
        "ue_policies": [
            {
                "supi": "imsi-208930000000009",
                "am_policy_associations": [{"id": "am-1"}],
                "ue_policy_associations": [
                    {
                        "id": "up-1",
                        "status": "complete",
                        "version": 4,
                        "policy": {
                            "upsc": 7,
                            "rules": [
                                {"precedence": 1, "routes": []},
                                {"precedence": 2, "routes": [{"dnn": "other"}]},
                            ],
                        },
                    }
                ],
            }
        ],
    }


def operation():
    return SimpleNamespace(
        resource_type="UrspRuleRequest",
        resource_id="ursp-app-f1",
        payload=UrspRuleRequest.model_validate(
            {
                "relatPrecedence": 1,
                "trafficDesc": {
                    "flowDescs": ["permit out tcp from any to 192.0.2.5 443"]
                },
                "routeSelParamSets": [
                    {
                        "precedence": 1,
                        "dnn": "embb",
                        "snssai": {"sst": 1, "sd": "000001"},
                    }
                ],
            }
        ),
    )


def test_ursp_preserves_other_rules_and_uses_real_version():
    result = project_policy_action(operation(), context(), "r1", "20893")
    assert result["parameters"]["associationId"] == "up-1"
    assert result["parameters"]["expectedVersion"] == 4
    assert result["parameters"]["policy"]["upsc"] == 7
    assert result["parameters"]["policy"]["rules"][0] == {
        "precedence": 2,
        "routes": [{"dnn": "other"}],
    }
    assert result["parameters"]["policy"]["rules"][1]["remoteIpv4"] == "192.0.2.5/32"


def test_cannot_overwrite_another_flows_precedence():
    op = operation()
    op.payload.relatPrecedence = 2
    with pytest.raises(ValueError, match="preserve"):
        project_policy_action(op, context(), "r1", "20893")


def test_domain_descriptor_is_explicitly_unsupported():
    op = operation()
    op.payload.trafficDesc.domainDescs = ["example.org"]
    with pytest.raises(ValueError, match="IPv4"):
        project_policy_action(op, context(), "r1", "20893")


def test_requires_completed_real_ue_association():
    native = context()
    native["ue_policies"][0]["ue_policy_associations"][0]["status"] = "delivered"
    with pytest.raises(ValueError, match="completed"):
        project_policy_action(operation(), native, "r1", "20893")


def test_am_uses_separate_association_and_ue_scope():
    op = SimpleNamespace(
        resource_type="PcfAmPolicyControlPolicyAssociation",
        resource_id="am-1",
        payload=PcfAmPolicyControlPolicyAssociation.model_validate(
            {
                "rfsp": 64,
                "suppFeat": "0",
                "request": {
                    "supi": "imsi-208930000000009",
                    "notificationUri": "http://unused.invalid",
                    "suppFeat": "0",
                },
            }
        ),
    )
    action = project_policy_action(op, context(), "r1", "20893")
    assert action["parameters"] == {"associationId": "am-1", "rfsp": 64}
    assert action["target"] == {"supi": "imsi-208930000000009"}
