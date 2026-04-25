import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain.control_plane import (
    ControlDomain,
    JointOptimizationRequest,
    MobilityContextSnapshot,
    OptimizationProblemConfig,
)
from agents.tools.optimizer.engine import SliceOptimizationEngine
from agents.tools.optimizer.joint_mobility import build_mobility_draft
from agents.tools.optimizer.models import (
    AMPolicyState,
    App,
    Flow,
    FlowAllocation,
    FlowService,
    FlowSLA,
    OptimizationConfig,
    Slice,
    SliceCapacity,
    SliceQos,
)
from model.PcfAmPolicyControl import (
    AccessType,
    Guami,
    PlmnIdNid,
    RatType,
    Snssai,
    UserLocation,
)


def test_problem_config_groups_two_domain_contract() -> None:
    config = OptimizationProblemConfig()

    assert config.grouped_decision_variables()["session_domain"] == [
        "slice_assignment",
        "bandwidth_allocation",
        "sm_policy_update",
        "ursp_update",
    ]
    assert "allowed_snssais" in config.grouped_decision_variables()["mobility_domain"]
    assert "snssai_alignment" in config.grouped_constraints()["coupling"]

    qos_only = config.normalized_for_domains([ControlDomain.QOS])
    assert "allowed_snssais" not in qos_only.decision_variables
    assert "slice_assignment" in qos_only.decision_variables

    mobility_only = config.normalized_for_domains([ControlDomain.MOBILITY])
    assert "slice_assignment" not in mobility_only.decision_variables
    assert "allowed_snssais" in mobility_only.decision_variables


def test_mobility_draft_does_not_emit_pra_trigger_without_presence_context() -> None:
    request = JointOptimizationRequest(
        target_ues=["imsi-001"],
        requested_domains=[ControlDomain.MOBILITY],
        operation_intent={"mobility_triggers": ["PRA_CH"]},
    )
    snapshot = MobilityContextSnapshot(
        supi="imsi-001",
        accessType=AccessType._3_GPP_ACCESS,
        ratType=RatType.NR,
        userLoc=UserLocation(),
        guami=Guami(plmnId=PlmnIdNid(mcc="001", mnc="01"), amfId="000001"),
        servingPlmn=PlmnIdNid(mcc="001", mnc="01"),
        allowedSnssais=[Snssai(sst=1, sd="000001")],
        targetSnssais=[Snssai(sst=1, sd="000001")],
        currentRfsp=2,
    )

    draft = build_mobility_draft(request, "imsi-001", snapshot, qos_plan={})

    assert "PRA_CH" not in [item.value for item in draft.policy.triggers or []]
    assert "LOC_CH" in [item.value for item in draft.policy.triggers or []]


def test_joint_engine_reports_session_mobility_and_coupling_costs() -> None:
    flow = Flow(
        id="flow-0001",
        name="video",
        service=FlowService(service_type="eMBB", service_type_id=1),
        sla=FlowSLA(
            bandwidth_ul=2.0,
            bandwidth_dl=4.0,
            guaranteed_bandwidth_ul=1.0,
            guaranteed_bandwidth_dl=2.0,
            latency=20.0,
            jitter=5.0,
            loss_rate=0.01,
            priority=1,
        ),
        allocation=FlowAllocation(optimize_requested=True),
    )
    app = App(id="app-0001", name="video-app", supi="imsi-001", flows=[flow])
    target_slice = Slice(
        name="embb",
        sst=1,
        sd="000001",
        capacity=SliceCapacity(total_bandwidth_ul=20.0, total_bandwidth_dl=20.0),
        qos=SliceQos(latency=5.0, jitter=1.0, loss_rate=0.001),
    )
    state = AMPolicyState(
        old_allowed_snssais=[],
        old_target_snssais=[],
        old_rfsp=4,
        old_triggers=[],
        mobility_risk_score=0.9,
    )
    engine = SliceOptimizationEngine(
        OptimizationConfig(
            enable_am_optimization=True,
            am_policy_state=state,
            solver_time_limit=5,
        )
    )

    _, _, status, _, breakdown = engine.solve([app], [target_slice], [])

    assert status == "Optimal"
    assert breakdown["session_cost"] >= 0.0
    assert breakdown["mobility_cost"] >= 0.0
    assert breakdown["coupling_cost"] >= 0.0
    assert breakdown["mobility_risk_score"] == 0.9
    assert breakdown["am_solution"]["allowed_snssais"] == ["01000001"]
