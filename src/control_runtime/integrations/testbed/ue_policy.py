"""Translate only the explicit, NAS-backed testbed policy profile."""

from __future__ import annotations

import re
from ipaddress import IPv4Network


def policy_state(native: dict, supi: str) -> dict:
    rows = [row for row in native["ue_policies"] if row["supi"] == supi]
    if len(rows) != 1:
        raise ValueError("snapshot has no unique, NAS-backed UE policy state")
    return rows[0]


def project_policy_action(operation, native: dict, run_id: str, plmn: str) -> dict:
    payload = operation.payload
    if operation.resource_type == "PcfAmPolicyControlPolicyAssociation":
        if payload.request is None or payload.rfsp is None:
            raise ValueError("AM update requires SUPI context and an explicit RFSP")
        supi = payload.request.supi
        state = policy_state(native, supi)
        association = next(
            row
            for row in state["am_policy_associations"]
            if row["id"] == operation.resource_id
        )
        if set(payload.model_dump(exclude_none=True)) - {"request", "rfsp", "suppFeat"}:
            raise ValueError(
                "NAS testbed AM profile supports RFSP only; other AM fields are unsupported"
            )
        return {
            "run_id": run_id,
            "snapshot_id": native["snapshot_id"],
            "action_type": "am_policy_apply",
            "target": {"supi": supi},
            "parameters": {"associationId": association["id"], "rfsp": payload.rfsp},
        }
    matches = [
        row
        for row in native["flows"]
        if operation.resource_id == f"ursp-{row['app_id']}-{row['id']}"
    ]
    if len(matches) != 1:
        raise ValueError(
            "URSP must bind one declared flow; application-ID matching is unsupported"
        )
    flow = matches[0]
    if payload.relatPrecedence != flow["ue_policy_route"]["rule_precedence"]:
        raise ValueError("a flow update must preserve its current rule precedence")
    descriptor = payload.trafficDesc
    if descriptor is None or any(
        value is not None
        for key, value in descriptor.model_dump().items()
        if key != "flowDescs"
    ):
        raise ValueError(
            "UE profile supports IPv4 remote-address/protocol/port descriptors only"
        )
    if descriptor.flowDescs is None or len(descriptor.flowDescs) != 1:
        raise ValueError("one IP flow description is required")
    match = re.fullmatch(
        r"permit out (ip|tcp|udp|6|17) from any to ([0-9./]+)(?: (\d+))?",
        descriptor.flowDescs[0],
    )
    if match is None:
        raise ValueError(
            "flow description must be: permit out <ip|tcp|udp> from any to <IPv4[/prefix]> [port]"
        )
    protocol, address, port = match.groups()
    rule = {
        "precedence": payload.relatPrecedence,
        "remoteIpv4": str(IPv4Network(address)),
        "protocol": {"ip": 0, "tcp": 6, "udp": 17, "6": 6, "17": 17}[protocol],
        "remotePort": int(port) if port else 0,
        "routes": [],
    }
    for route in payload.routeSelParamSets:
        if route.snssai is None or route.snssai.sd is None:
            raise ValueError("this UE profile requires explicit SST and SD")
        rule["routes"].append(
            {
                "precedence": route.precedence,
                "dnn": route.dnn,
                "sst": route.snssai.sst,
                "sd": route.snssai.sd,
            }
        )
    state = policy_state(native, flow["supi"])
    associations = state["ue_policy_associations"]
    if len(associations) != 1 or associations[0]["status"] != "complete":
        raise ValueError("a completed initial UE policy is required")
    association = associations[0]
    rules = [
        row
        for row in association["policy"]["rules"]
        if row["precedence"] != rule["precedence"]
    ]
    rules.append(rule)
    return {
        "run_id": run_id,
        "snapshot_id": native["snapshot_id"],
        "action_type": "ursp_apply",
        "target": {"supi": flow["supi"], "flow_id": flow["id"]},
        "parameters": {
            "associationId": association["id"],
            "expectedVersion": association["version"],
            "policy": {
                "plmn": plmn,
                "upsc": association["policy"]["upsc"],
                "rules": rules,
            },
        },
    }
