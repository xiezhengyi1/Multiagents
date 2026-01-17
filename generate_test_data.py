from model.UeContext import UeSmPolicyData, SmPolicyData
from model.SmPolicyContextData import SmPolicyContextData, Snssai, PduSessionType, SubscribedDefaultQos, VplmnQos
from model.UserLocation import UserLocation, NrLocation, GlobalRanNodeId, Tai, PlmnId, Ncgi
from model.Arp import Arp

def create_test_ue_sm_policy_data():
    # 1. Create SmPolicyContextData
    # constructing a sample Snssai
    snssai = Snssai(sst=1, sd="000001")
    
    # constructing a sample UserLocation
    plmn_id = PlmnId(mcc="460", mnc="00")
    tai = Tai(plmnId=plmn_id, tac="000001")
    
    ncgi = Ncgi(plmnId=plmn_id, nrCellId="000000001")
    nr_loc = NrLocation(tai=tai, ncgi=ncgi)
    
    user_loc = UserLocation(nrLocation=nr_loc)

    # SubscribedDefaultQos
    subs_def_qos = SubscribedDefaultQos(
        **{"5qi": 9},
        arp=Arp(priorityLevel=1, preemptCap="NOT_PREEMPT", preemptVuln="NOT_PREEMPTABLE"),
        priorityLevel=1
    )

    # VplmnQos
    vplmn_qos = VplmnQos(
        **{"5qi": 9},
        arp=Arp(priorityLevel=1, preemptCap="NOT_PREEMPT", preemptVuln="NOT_PREEMPTABLE"),
        maxFbrDl="100 Mbps",
        maxFbrUl="50 Mbps"
    )

    policy_context = SmPolicyContextData(
        supi="imsi-123456789012345",
        pduSessionId=1,
        dnn="internet",
        sliceInfo=snssai,
        pduSessionType=PduSessionType.IPV4,
        notificationUri="http://example.com/notify",
        userLocationInfo=user_loc,
        subsDefQos=subs_def_qos,
        vplmnQos=vplmn_qos
    )

    # 2. Create SmPolicyData
    # mock some snssai data
    # Note: SmPolicySnssaiData might be needed but not imported or used correctly in original script if not careful
    # Assuming SmPolicyData structure is simple dict or specific model. 
    # Checking UeContext.py content previously, SmPolicyData was imported but definition not fully visible.
    # Assuming the user's previous code structure was roughly correct on fields.
    
    sm_policy_data = SmPolicyData(
        smPolicySnssaiData={
            "01000001": {
                "snssai": {"sst": 1, "sd": "000001"},
                "smPolicyDnnData": {"internet": {"gbrUl": "100 Mbps", "gbrDl": "200 Mbps"}}
            }
        }
        # suppFeat="feature1" # removed unless sure it exists in SmPolicyData definition
    )

    # 3. Create UeSmPolicyData
    ue_sm_policy_data = UeSmPolicyData(
        remainGbrUL=50.0,
        remainGbrDL=100.0,
        smPolicyData=sm_policy_data,
        policyContext=policy_context
    )

    return ue_sm_policy_data

if __name__ == "__main__":
    data = create_test_ue_sm_policy_data()
    print("Successfully created UeSmPolicyData:")
    print(data.model_dump_json(indent=2, exclude_none=True))
