from __future__ import annotations

from typing import TYPE_CHECKING

from control_runtime.simulation import ExecutionAction, ExecutionOperation
from model.PcfAmPolicyControl import (
    Ambr,
    PcfAmPolicyControlPolicyAssociation,
    PcfAmPolicyControlPolicyAssociationRequest,
    PcfAmPolicyControlRequestTrigger,
    Snssai,
)
from model.SmPolicyDecision import (
    FlowDirection,
    FlowInformation,
    PCCRule,
    QosData,
    SmPolicyDecision,
)
from model.UrspRuleRequest import (
    AppDescriptor,
    RouteSelectionParameterSet,
    TrafficDescriptorComponents,
    UrspRuleRequest,
)
from model.UrspRuleRequest import Snssai as UrspSnssai

if TYPE_CHECKING:
    from control_runtime.actors.ppa import PpaProposal, SnssaiSpec
    from control_runtime.actors.sba import SbaArtifact


def _snssai_payload(spec: SnssaiSpec) -> dict[str, int | str]:
    payload: dict[str, int | str] = {"sst": spec.sst}
    if spec.sd is not None:
        payload["sd"] = spec.sd
    return payload


def compile_ppa_execution_action(
    *,
    proposal: PpaProposal,
    sba_artifact: SbaArtifact,
) -> ExecutionAction:
    operations: list[ExecutionOperation] = []
    if proposal.am_policy is not None:
        if sba_artifact.mobility_intent is None:
            raise ValueError("AM policy requires an SBA mobility binding")
        am = proposal.am_policy
        ue_ambr = (
            Ambr(
                uplink=f"{am.ue_ambr_ul_mbps:g} Mbps",
                downlink=f"{am.ue_ambr_dl_mbps:g} Mbps",
            )
            if am.ue_ambr_ul_mbps is not None and am.ue_ambr_dl_mbps is not None
            else None
        )
        request = PcfAmPolicyControlPolicyAssociationRequest(
            notificationUri=am.notification_uri,
            supi=sba_artifact.supi,
            rfsp=am.rfsp,
            ueAmbr=ue_ambr,
            allowedSnssais=[
                Snssai(**_snssai_payload(item)) for item in am.allowed_snssais
            ],
            targetSnssais=[
                Snssai(**_snssai_payload(item)) for item in am.target_snssais
            ],
            suppFeat=am.supp_feat,
        )
        association = PcfAmPolicyControlPolicyAssociation(
            request=request,
            triggers=[PcfAmPolicyControlRequestTrigger(item) for item in am.triggers],
            rfsp=am.rfsp,
            ueAmbr=ue_ambr,
            suppFeat=am.supp_feat,
        )
        operations.append(
            ExecutionOperation(
                operation_id=f"op-{len(operations) + 1:03d}",
                domain="mobility",
                operation="update",
                resource_type="PcfAmPolicyControlPolicyAssociation",
                resource_id=sba_artifact.mobility_intent.association_id,
                payload=association,
            )
        )

    for policy in proposal.sm_policies:
        qos_id = f"qos-{policy.flow_id}"
        pcc_id = f"pcc-{policy.flow_id}"
        qos_payload = QosData(
            qosId=qos_id,
            priorityLevel=policy.priority,
            packetDelayBudget=int(policy.target_latency_ms),
            packetErrorRate=str(policy.packet_error_rate),
            maxbrUl=f"{policy.max_br_ul_mbps:g} Mbps",
            maxbrDl=f"{policy.max_br_dl_mbps:g} Mbps",
            gbrUl=(
                f"{policy.gbr_ul_mbps:g} Mbps"
                if policy.gbr_ul_mbps is not None
                else None
            ),
            gbrDl=(
                f"{policy.gbr_dl_mbps:g} Mbps"
                if policy.gbr_dl_mbps is not None
                else None
            ),
        )
        sm_decision = SmPolicyDecision(
            pccRules={
                pcc_id: PCCRule(
                    pccRuleId=pcc_id,
                    precedence=policy.priority,
                    refQosData=[qos_id],
                    appId=policy.app_id,
                    flowInfos=[
                        FlowInformation(
                            flowDirection=FlowDirection.BIDIRECTIONAL,
                            flowDescription=policy.flow_description,
                        )
                    ],
                )
            },
            qosDecs={qos_id: qos_payload},
        )
        operations.append(
            ExecutionOperation(
                operation_id=f"op-{len(operations) + 1:03d}",
                domain="qos",
                operation="update",
                resource_type="SmPolicyDecision",
                resource_id=f"smp-{policy.app_id}-{policy.flow_id}",
                payload=sm_decision,
            )
        )

    for index, policy in enumerate(proposal.ursp_policies, start=1):
        descriptor = None
        if policy.traffic_desc is not None:
            spec = policy.traffic_desc
            descriptor = TrafficDescriptorComponents(
                flowDescs=list(spec.flow_descs) or None,
                dnns=list(spec.dnns) or None,
                appDescs=(
                    {
                        spec.os_id: AppDescriptor(
                            osId=spec.os_id,
                            appIds={app_id: app_id for app_id in spec.app_ids},
                        )
                    }
                    if spec.os_id is not None and spec.app_ids
                    else None
                ),
            )
        ursp_request = UrspRuleRequest(
            trafficDesc=descriptor,
            relatPrecedence=policy.relat_precedence,
            routeSelParamSets=[
                RouteSelectionParameterSet(
                    dnn=route.dnn,
                    precedence=route.precedence,
                    snssai=(
                        UrspSnssai(**_snssai_payload(route.snssai))
                        if route.snssai is not None
                        else None
                    ),
                )
                for route in policy.route_sel_param_sets
            ],
        )
        operations.append(
            ExecutionOperation(
                operation_id=f"op-{len(operations) + 1:03d}",
                domain="qos",
                operation="update",
                resource_type="UrspRuleRequest",
                resource_id=f"ursp-{policy.app_id}-{policy.flow_id or index}",
                payload=ursp_request,
            )
        )
    return ExecutionAction(operations=operations)


__all__ = ["compile_ppa_execution_action"]
