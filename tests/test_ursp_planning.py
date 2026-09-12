from __future__ import annotations

from pathlib import Path

from control_runtime.actors.ppa import PpaProposal, validate_ppa_proposal
from control_runtime.actors.sba import SbaArtifact
from shared.runtime.journal import EpisodeJournal


def _artifact(*, pdu_session_pool: list[dict[str, object]] | None = None) -> SbaArtifact:
    return SbaArtifact.model_validate(
        {
            "supi": "imsi-208930000000001",
            "flows": [
                {
                    "app_id": "app-video",
                    "flow_id": "flow-1",
                    "name": "video",
                    "resolution_status": "resolved",
                }
            ],
            "mobility_intent": None,
            "qos_operation_constraints": [],
            "slice_migration_authorization": {
                "decision": "not_applicable",
                "target_snssais": [],
                "subscription_change_required": False,
                "rationale": ["route is authorized"],
            },
            "open_questions": [],
            "subscription": {
                "allowed_snssais": ["01010203"],
                "session_management": [
                    {"snssai": "01010203", "dnn_configurations": {"embb": {}}}
                ],
            },
            "flow_catalog": [
                {
                    "flow_id": "flow-1",
                    "app_id": "app-video",
                    "traffic": {
                        "five_tuple": ["10.0.0.1", "192.0.2.5", 30000, 443, "tcp"]
                    },
                    "ue_policy_route": {"rule_precedence": 7},
                }
            ],
            "ue_policy_associations": [
                {"id": "ue-policy-1", "status": "complete", "version": 4}
            ],
            "pdu_session_pool": pdu_session_pool
            if pdu_session_pool is not None
            else [
                {
                    "pdu_session_id": 2,
                    "dnn": "embb",
                    "snssai": "01010203",
                }
            ],
        }
    )


def _proposal(*, dnn: str = "embb") -> PpaProposal:
    return PpaProposal.model_validate(
        {
            "planning_status": "executable_plan",
            "rationale": "route the grounded video flow to its authorized PDU session",
            "missing_evidence": [],
            "blocked_targets": [],
            "upstream_requests": [],
            "planner_conflicts": [],
            "open_questions": [],
            "sm_policies": [],
            "am_policy": None,
            "ursp_policies": [
                {
                    "target_type": "flow",
                    "app_id": "app-video",
                    "flow_id": "flow-1",
                    "relat_precedence": 7,
                    "traffic_desc": {
                        "flow_descs": [
                            "permit out tcp from any to 192.0.2.5 443"
                        ],
                        "dnns": [],
                        "app_ids": [],
                        "os_id": None,
                    },
                    "route_sel_param_sets": [
                        {
                            "dnn": dnn,
                            "precedence": 1,
                            "snssai": {"sst": 1, "sd": "010203"},
                        }
                    ],
                }
            ],
            "partial_policies": [],
        }
    )


def _issues(
    *, proposal: PpaProposal, artifact: SbaArtifact, tmp_path: Path
) -> list[str]:
    journal = EpisodeJournal(root=tmp_path, episode_id="ursp-validator")
    return [
        issue.code
        for issue in validate_ppa_proposal(
            proposal=proposal,
            sba_artifact=artifact,
            target_agent="ppa_compiler",
            message_type="proposal",
            trigger_message_type="proposal",
            observation_queries=set(),
            journal=journal,
            input_artifact_ids=[],
        )
    ]


def test_ursp_accepts_only_catalog_subscription_and_pdu_evidence(
    tmp_path: Path,
) -> None:
    assert _issues(
        proposal=_proposal(), artifact=_artifact(), tmp_path=tmp_path
    ) == []


def test_ursp_rejects_an_unsubscribed_route(tmp_path: Path) -> None:
    assert "unsubscribed_ursp_route" in _issues(
        proposal=_proposal(dnn="urllc"), artifact=_artifact(), tmp_path=tmp_path
    )


def test_ursp_requires_the_selected_route_in_the_existing_pdu_pool(
    tmp_path: Path,
) -> None:
    assert "dynamic_pdu_required" in _issues(
        proposal=_proposal(), artifact=_artifact(pdu_session_pool=[]), tmp_path=tmp_path
    )
