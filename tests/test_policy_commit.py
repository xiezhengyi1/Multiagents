from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)

from control_runtime.integrations.storage import ue_store


SUPI = "imsi-001010123456789"
SESSION_ID = "session-1"
SNAPSHOT_ID = "snapshot-1"


def _sm_policy() -> dict[str, Any]:
    return {
        "policy_id": "smp-video-flow-1",
        "policy_type": "SmPolicyDecision",
        "policy_details": {
            "pccRules": {"pcc-flow-1": {"flow_id": "flow-1"}},
            "qosDecs": {"qos-flow-1": {"qosId": "qos-flow-1"}},
            "sessRules": {"sess-flow-1": {"authDefQos": {"5qi": 7}}},
        },
    }


def _ursp_policy() -> dict[str, Any]:
    return {
        "policy_id": "ursp-video-flow-1",
        "policy_type": "UrspRuleRequest",
        "policy_details": {
            "routeSelParamSets": [{"routeSelDesc": "video"}],
            "trafficDesc": {"flow_id": "flow-1"},
        },
    }


def _am_policy() -> dict[str, Any]:
    return {
        "policy_id": "am-policy-1",
        "policy_type": ue_store.AM_POLICY_TYPE,
        "policy_details": {
            "request": {
                "supi": SUPI,
                "allowedSnssais": [{"sst": 1, "sd": "010203"}],
                "targetSnssais": [{"sst": 2, "sd": "040506"}],
                "mappingSnssais": [{"servingSnssai": {"sst": 1}}],
                "rfsp": 9,
                "ueAmbr": {"uplink": "10 Mbps", "downlink": "20 Mbps"},
            },
            "triggers": ["LOC_CH"],
            "ueSliceMbrs": [{"sliceMbr": {"up": "1 Mbps"}}],
            "pras": {"pra-1": {"praId": "1"}},
        },
    }


def test_commit_access_policies_merges_qos_and_ursp_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = {
        "smPolicyData": {"old-sm": {"keep": True}},
        "pccRules": {"old-sm": {"old-pcc": {}}},
        "qosDecs": {"old-sm": {"old-qos": {}}},
        "sessRules": {"old-sm": {"old-sess": {}}},
        "traffContDecs": {"old-sm": {"old-tc": {}}},
        "chgDecs": {"old-sm": {"old-chg": {}}},
        "urspRules": {"old-ursp": {"keep": True}},
        "app_catalog": [{"id": "stale-app"}],
        "flow_catalog": [{"id": "stale-flow"}],
    }
    catalog = {
        "app_catalog": [{"id": "app-1", "supi": SUPI}],
        "flow_catalog": [{"id": "flow-1", "app_id": "app-1"}],
    }
    upserts: list[dict[str, Any]] = []

    def load_context(supi: str, snapshot_id: str = "") -> dict[str, Any]:
        assert (supi, snapshot_id) == (SUPI, SNAPSHOT_ID)
        return existing

    def load_catalog(supi: str, snapshot_id: str = "") -> dict[str, Any]:
        assert (supi, snapshot_id) == (SUPI, SNAPSHOT_ID)
        return catalog

    def persist_context(**kwargs: Any) -> bool:
        upserts.append(kwargs)
        return True

    monkeypatch.setattr(ue_store, "get_ue_context_by_supi", load_context)
    monkeypatch.setattr(ue_store, "get_ue_flow_catalog_by_supi", load_catalog)
    monkeypatch.setattr(ue_store, "upsert_ue_context", persist_context)

    sm_policy = _sm_policy()
    ursp_policy = _ursp_policy()
    ue_store.commit_applied_policy_plan(
        supi=SUPI,
        policies=[sm_policy, ursp_policy],
        session_id=SESSION_ID,
        snapshot_id=SNAPSHOT_ID,
    )

    assert len(upserts) == 1
    persisted = upserts[0]
    assert persisted["supi"] == SUPI
    assert persisted["sm_policy_data"] == {
        "old-sm": {"keep": True},
        "smp-video-flow-1": {
            "policy_id": "smp-video-flow-1",
            "policy_type": "SmPolicyDecision",
            "policy_details": sm_policy["policy_details"],
        },
    }
    assert persisted["pcc_rules"]["smp-video-flow-1"] == sm_policy["policy_details"]["pccRules"]
    assert persisted["qos_decs"]["smp-video-flow-1"] == sm_policy["policy_details"]["qosDecs"]
    assert persisted["sess_rules"]["smp-video-flow-1"] == sm_policy["policy_details"]["sessRules"]
    assert persisted["ursp_rules"] == {
        "old-ursp": {"keep": True},
        "ursp-video-flow-1": ursp_policy["policy_details"],
    }
    assert persisted["app_catalog"] == catalog["app_catalog"]
    assert persisted["flow_catalog"] == catalog["flow_catalog"]


def test_commit_am_policy_persists_association_mobility_context_and_amf_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = {
        "amPolicyContext": {"associations": {"old-am": {"keep": True}}, "retained": "value"},
        "accessMobilityContext": {"access_type": "3GPP_ACCESS"},
        "servingNfContext": {"amf_id": "amf-1", "amf_uri": "https://amf.example.test"},
    }
    association_upserts: list[dict[str, Any]] = []
    context_upserts: list[dict[str, Any]] = []
    nf_bindings: list[dict[str, Any]] = []
    mobility_events: list[dict[str, Any]] = []

    monkeypatch.setattr(ue_store, "get_ue_context_by_supi", lambda supi, snapshot_id="": existing)
    monkeypatch.setattr(
        ue_store,
        "upsert_am_policy_association",
        lambda **kwargs: association_upserts.append(kwargs) or True,
    )
    monkeypatch.setattr(ue_store, "upsert_ue_context", lambda **kwargs: context_upserts.append(kwargs) or True)
    monkeypatch.setattr(ue_store, "upsert_serving_nf_binding", lambda **kwargs: nf_bindings.append(kwargs) or True)
    monkeypatch.setattr(ue_store, "record_mobility_event", lambda **kwargs: mobility_events.append(kwargs) or True)

    policy = _am_policy()
    ue_store.commit_applied_policy_plan(
        supi=SUPI,
        policies=[policy],
        session_id=SESSION_ID,
        snapshot_id=SNAPSHOT_ID,
    )

    assert association_upserts == [
        {
            "supi": SUPI,
            "pol_asso_id": "am-policy-1",
            "association_request": policy["policy_details"]["request"],
            "association_policy": policy["policy_details"],
            "status": "applied",
            "trigger_event": "JOINT_CONTROL_REEVALUATION",
            "session_id": SESSION_ID,
            "snapshot_id": SNAPSHOT_ID,
            "round_index": 1,
        }
    ]
    assert len(context_upserts) == 1
    persisted_context = context_upserts[0]
    assert persisted_context["supi"] == SUPI
    assert persisted_context["access_mobility_context"] == existing["accessMobilityContext"]
    assert persisted_context["serving_nf_context"] == existing["servingNfContext"]
    assert persisted_context["am_policy_context"] == {
        "associations": {"old-am": {"keep": True}, "am-policy-1": policy["policy_details"]},
        "retained": "value",
        "allowedSnssais": policy["policy_details"]["request"]["allowedSnssais"],
        "targetSnssais": policy["policy_details"]["request"]["targetSnssais"],
        "mappingSnssais": policy["policy_details"]["request"]["mappingSnssais"],
        "rfsp": 9,
        "ueAmbr": policy["policy_details"]["request"]["ueAmbr"],
        "ueSliceMbrs": policy["policy_details"]["ueSliceMbrs"],
        "pras": policy["policy_details"]["pras"],
    }
    assert persisted_context["mobility_summary"] == {
        "currentAssociationId": "am-policy-1",
        "currentTriggers": ["LOC_CH"],
        "currentRfsp": 9,
        "lastUpdatedReason": "policy_dispatch",
    }
    assert nf_bindings == [
        {
            "supi": SUPI,
            "nf_type": "AMF",
            "nf_instance_id": "amf-1",
            "nf_uri": "https://amf.example.test",
            "binding_info": existing["servingNfContext"],
        }
    ]
    assert mobility_events == [
        {
            "supi": SUPI,
            "session_id": SESSION_ID,
            "snapshot_id": SNAPSHOT_ID,
            "event_type": "JOINT_CONTROL_REEVALUATION",
            "event_summary": "AM policy association updated from policy dispatch execution",
            "event_payload": {
                "association_id": "am-policy-1",
                "request": policy["policy_details"]["request"],
                "policy": policy["policy_details"],
            },
        }
    ]


def test_commit_access_policy_stops_when_context_persistence_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    association_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(ue_store, "get_ue_context_by_supi", lambda supi, snapshot_id="": {})
    monkeypatch.setattr(
        ue_store,
        "get_ue_flow_catalog_by_supi",
        lambda supi, snapshot_id="": {"app_catalog": [], "flow_catalog": []},
    )
    monkeypatch.setattr(ue_store, "upsert_ue_context", lambda **kwargs: False)
    monkeypatch.setattr(
        ue_store,
        "upsert_am_policy_association",
        lambda **kwargs: association_calls.append(kwargs) or True,
    )

    with pytest.raises(RuntimeError, match=f"failed to persist policies for supi {SUPI}"):
        ue_store.commit_applied_policy_plan(
            supi=SUPI,
            policies=[_sm_policy(), _am_policy()],
            session_id=SESSION_ID,
            snapshot_id=SNAPSHOT_ID,
        )

    assert association_calls == []
