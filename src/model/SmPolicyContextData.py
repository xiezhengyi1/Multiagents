from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from .Arp import Arp
from .RatType import RatType
from .UserLocation import (
    Ambr,
    Ecgi,
    GlobalRanNodeId,
    Ncgi,
    Tai,
    UserLocation,
)

# --- Sm Policy Context Data Definitions ---

class SubscribedDefaultQos(BaseModel):
    var5qi: int = Field(..., alias="5qi")
    arp: Arp | None = None
    priorityLevel: int | None = None

class VplmnQos(BaseModel):
    var5qi: int | None = Field(None, alias="5qi")
    arp: Arp | None = None
    sessionAmbr: Ambr | None = None
    maxFbrDl: str | None = None
    maxFbrUl: str | None = None
    guaFbrDl: str | None = None
    guaFbrUl: str | None = None

class AccuUsageReports(BaseModel):
    refUmIds: str
    volUsage: int | None = None
    volUsageUplink: int | None = None
    volUsageDownlink: int | None = None
    timeUsage: int | None = None
    nextVolUsage: int | None = None
    nextVolUsageUplink: int | None = None
    nextVolUsageDownlink: int | None = None
    nextTimeUsage: int | None = None

class PresenceState(str, Enum):
    IN_AREA = "IN_AREA"
    OUT_OF_AREA = "OUT_OF_AREA"
    UNKNOWN = "UNKNOWN"
    INACTIVE = "INACTIVE"

class RequestQos(BaseModel):
    var5qi: int
    gbrUl: str | None = None
    gbrDl: str | None = None

class RuleOperation(str, Enum):
    CREATE_PCC_RULE = "CREATE_PCC_RULE"
    DELETE_PCC_RULE = "DELETE_PCC_RULE"
    MODIFY_PCC_RULE_AND_ADD_PACKET_FILTERS = "MODIFY_PCC_RULE_AND_ADD_PACKET_FILTERS"
    MODIFY_PCC_RULE_AND_REPLACE_PACKET_FILTERS = "MODIFY_PCC_RULE_AND_REPLACE_PACKET_FILTERS"
    MODIFY_PCC_RULE_AND_DELETE_PACKET_FILTERS = "MODIFY_PCC_RULE_AND_DELETE_PACKET_FILTERS"
    MODIFY_PCC_RULE_WITHOUT_MODIFY_PACKET_FILTERS = "MODIFY_PCC_RULE_WITHOUT_MODIFY_PACKET_FILTERS"

class FlowDirection(str, Enum):
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    UNSPECIFIED = "UNSPECIFIED"

class AccNetChId(BaseModel):
    accNetChaIdValue: int | None = None
    accNetChargIdString: str | None = None
    refPccRuleIds: list[str] | None = None
    sessionChScope: bool | None = None

class AccessType(str, Enum):
    _3GPP_ACCESS = "3GPP_ACCESS"
    NON_3GPP_ACCESS = "NON_3GPP_ACCESS"

class AdditionalAccessInfo(BaseModel):
    accessType: list[AccessType] | None = None
    ratType: RatType | None = None

class PlmnIdNid(BaseModel):
    mcc: str
    mnc: str
    nid: str | None = None

class EthFlowDescription(BaseModel):
    destMacAddr: str | None = None
    ethType: str
    fDesc: str | None = None
    fDir: FlowDirection | None = None
    sourceMacAddr: str | None = None
    vlanTags: list[str] | None = None
    srcMacAddrEnd: str | None = None
    destMacAddrEnd: str | None = None

class FlowInformation(BaseModel):
    flowDescription: str | None = None
    ethFlowDescription: EthFlowDescription | None = None
    packFiltId: str | None = None
    packetFilterUsage: bool | None = None
    tosTrafficClass: str | None = None
    spi: str | None = None
    flowLabel: str | None = None
    flowDirection: FlowDirection | None = None

class AppDetectionInfo(BaseModel):
    appId: str
    instanceId: str | None = None
    sdfDescriptions: list[FlowInformation] | None = None

class RuleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

class PcfSmPolicyControlFailureCode(str, Enum):
    UNK_RULE_ID = "UNK_RULE_ID"
    RA_GR_ERR = "RA_GR_ERR"
    SER_ID_ERR = "SER_ID_ERR"
    NF_MAL = "NF_MAL"
    RES_LIM = "RES_LIM"
    MAX_NR_QoS_FLOW = "MAX_NR_QoS_FLOW"
    MISS_FLOW_INFO = "MISS_FLOW_INFO"
    RES_ALLO_FAIL = "RES_ALLO_FAIL"
    UNSUCC_QOS_VAL = "UNSUCC_QOS_VAL"
    INCOR_FLOW_INFO = "INCOR_FLOW_INFO"
    PS_TO_CS_HAN = "PS_TO_CS_HAN"
    APP_ID_ERR = "APP_ID_ERR"
    NO_QOS_FLOW_BOUND = "NO_QOS_FLOW_BOUND"
    FILTER_RES = "FILTER_RES"
    MISS_REDI_SER_ADDR = "MISS_REDI_SER_ADDR"
    CM_END_USER_SER_DENIED = "CM_END_USER_SER_DENIED"
    CM_CREDIT_CON_NOT_APP = "CM_CREDIT_CON_NOT_APP"
    CM_AUTH_REJ = "CM_AUTH_REJ"
    CM_USER_UNK = "CM_USER_UNK"
    CM_RAT_FAILED = "CM_RAT_FAILED"
    UE_STA_SUSP = "UE_STA_SUSP"
    UNKNOWN_REF_ID = "UNKNOWN_REF_ID"
    INCORRECT_COND_DATA = "INCORRECT_COND_DATA"
    REF_ID_COLLISION = "REF_ID_COLLISION"
    TRAFFIC_STEERING_ERROR = "TRAFFIC_STEERING_ERROR"
    DNAI_STEERING_ERROR = "DNAI_STEERING_ERROR"
    AN_GW_FAILE = "AN_GW_FAILE"
    MAX_NR_PACKET_FILTERS_EXCEEDED = "MAX_NR_PACKET_FILTERS_EXCEEDED"
    PACKET_FILTER_TFT_ALLOCATION_EXCEEDED = "PACKET_FILTER_TFT_ALLOCATION_EXCEEDED"
    MUTE_CHG_NOT_ALLOWED = "MUTE_CHG_NOT_ALLOWED"

class FinalUnitAction(str, Enum):
    TERMINATE = "TERMINATE"
    REDIRECT = "REDIRECT"
    RESTRICT_ACCESS = "RESTRICT_ACCESS"

class NgApCause(BaseModel):
    group: int
    value: int

class RanNasRelCause(BaseModel):
    ngApCause: NgApCause | None = None
    var5gMmCause: int | None = Field(None, alias="5gMmCause")
    var5gSmCause: int | None = Field(None, alias="5gSmCause")
    epsCause: str | None = None

class RuleReport(BaseModel):
    pccRuleIds: list[str]
    ruleStatus: list[RuleStatus]
    contVers: list[int] | None = None
    failureCode: list[PcfSmPolicyControlFailureCode] | None = None
    finUnitAct: list[FinalUnitAction] | None = None
    ranNasRelCauses: list[RanNasRelCause] | None = None
    altQosParamId: str | None = None

class SessionRuleFailureCode(str, Enum):
    NF_MAL = "NF_MAL"
    RES_LIM = "RES_LIM"
    SESSION_RESOURCE_ALLOCATION_FAILURE = "SESSION_RESOURCE_ALLOCATION_FAILURE"
    UNSUCC_QOS_VAL = "UNSUCC_QOS_VAL"
    INCORRECT_UM = "INCORRECT_UM"
    UE_STA_SUSP = "UE_STA_SUSP"
    UNKNOWN_REF_ID = "UNKNOWN_REF_ID"
    INCORRECT_COND_DATA = "INCORRECT_COND_DATA"
    REF_ID_COLLISION = "REF_ID_COLLISION"
    AN_GW_FAILED = "AN_GW_FAILED"

class SessionRuleReport(BaseModel):
    ruleIds: list[str]
    ruleStatus: list[RuleStatus]
    sessRuleFailureCode: list[SessionRuleFailureCode] | None = None

class QosNotifType(str, Enum):
    GUARANTEED = "GUARANTEED"
    NOT_GUARANTEED = "NOT_GUARANTEED"

class PcfSmPolicyControlQosNotificationControlInfo(BaseModel):
    refPccRuleIds: list[str]
    notifType: list[QosNotifType]
    contVer: int | None = None
    altQosParamId: str | None = None

class PcfSmPolicyControlQosMonitoringReport(BaseModel):
    refPccRuleIds: list[str]
    ulDelays: list[int] | None = None
    dlDelays: list[int] | None = None
    rtDelays: list[int] | None = None
    pdmf: bool | None = None

class PresenceInfo(BaseModel):
    praId: str | None = None
    fDir: FlowDirection | None = None
    presenceState: PresenceState | None = None
    trackingAreaList: list[Tai] | None = None
    ecgiList: list[Ecgi] | None = None
    ncgiList: list[Ncgi] | None = None
    globalRanNodeIdList: list[GlobalRanNodeId] | None = None
    globaleNbIdList: list[GlobalRanNodeId] | None = None

class PacketFilterInfo(BaseModel):
    packFiltId: str | None = None
    packFiltCont: str | None = None
    tosTrafficClass: str | None = None
    spi: str | None = None
    flowLabel: str | None = None
    flowDirection: FlowDirection | None = None

class UeInitiatedResourceRequest(BaseModel):
    pccRuleId: str | None = None
    ruleOp: RuleOperation
    precedence: int | None = None
    packFiltInfo: list[PacketFilterInfo]
    reqQos: RequestQos | None = None

class QosFlowUsage(str, Enum):
    GENERAL = "GENERAL"
    IMS_SIG = "IMS_SIG"

class CreditManagementStatus(str, Enum):
    END_USER_SER_DENIED = "END_USER_SER_DENIED"
    CREDIT_CTRL_NOT_APP = "CREDIT_CTRL_NOT_APP"
    AUTH_REJECTED = "AUTH_REJECTED"
    USER_UNKNOWN = "USER_UNKNOWN"
    RATING_FAILED = "RATING_FAILED"

class Guami(BaseModel):
    plmnId: PlmnIdNid
    amfId: str

class AnGwAddress(BaseModel):
    anGwIpv4Addr: str | None = None
    anGwIpv6Addr: str | None = None

class SgsnAddress(BaseModel):
    sgsnIpv4Addr: str | None = None
    sgsnIpv6Addr: str | None = None

class TraceDepth(str, Enum):
    MINIMUM = "MINIMUM"
    MEDIUM = "MEDIUM"
    MAXIMUM = "MAXIMUM"
    MINIMUM_WO_VENDOR_EXTENSION = "MINIMUM_WO_VENDOR_EXTENSION"
    MEDIUM_WO_VENDOR_EXTENSION = "MEDIUM_WO_VENDOR_EXTENSION"
    MAXIMUM_WO_VENDOR_EXTENSION = "MAXIMUM_WO_VENDOR_EXTENSION"

class TraceData(BaseModel):
    traceRef: str
    traceDepth: TraceDepth
    neTypeList: str | None = None
    eventList: str | None = None
    collectionEntityIpv4Addr: str | None = None
    collectionEntityIpv6Addr: str | None = None
    interfaceList: str | None = None

class ServingNfIdentity(BaseModel):
    servNfInstId: str | None = None
    guami: Guami | None = None
    anGwAddr: AnGwAddress | None = None
    sgsnAddr: SgsnAddress | None = None

class MaPduIndication(str, Enum):
    MA_PDU_REQUEST = "MA_PDU_REQUEST"
    MA_PDU_NETWORK_UPGRADE_ALLOWED = "MA_PDU_NETWORK_UPGRADE_ALLOWED"

class PcfSmPolicyControlAtsssCapability(str, Enum):
    MPTCP_ATSSS_LL_WITH_ASMODE_UL = "MPTCP_ATSSS_LL_WITH_ASMODE_UL"
    MPTCP_ATSSS_LL_WITH_EXSDMODE_DL_ASMODE_UL = "MPTCP_ATSSS_LL_WITH_EXSDMODE_DL_ASMODE_UL"
    MPTCP_ATSSS_LL_WITH_ASMODE_DLUL = "MPTCP_ATSSS_LL_WITH_ASMODE_DLUL"
    ATSSS_LL = "ATSSS_LL"
    MPTCP_ATSSS_LL = "MPTCP_ATSSS_LL"

class TsnBridgeInfo(BaseModel):
    bridgeId: int | None = None
    dsttAddr: str | None = None
    dsttPortNum: int | None = None
    dsttResidTime: int | None = None

class BridgeManagementContainer(BaseModel):
    bridgeManCont: str

class PortManagementContainer(BaseModel):
    portManCont: str
    portNum: int

class IpMulticastAddressInfo(BaseModel):
    srcIpv4Addr: str | None = None
    ipv4MulAddr: str | None = None
    srcIpv6Addr: str | None = None
    ipv6MulAddr: str | None = None

class PolicyDecisionFailureCode(str, Enum):
    TRA_CTRL_DECS_ERR = "TRA_CTRL_DECS_ERR"
    QOS_DECS_ERR = "QOS_DECS_ERR"
    CHG_DECS_ERR = "CHG_DECS_ERR"
    USA_MON_DECS_ERR = "USA_MON_DECS_ERR"
    QOS_MON_DECS_ERR = "QOS_MON_DECS_ERR"
    CON_DATA_ERR = "CON_DATA_ERR"
    POLICY_PARAM_ERR = "POLICY_PARAM_ERR"

class InvalidParam(BaseModel):
    param: str
    reason: str | None = None

class DddTrafficDescriptor(BaseModel):
    ipv4Addr: str | None = None
    ipv6Addr: str | None = None
    portNumber: int | None = None
    macAddr: str | None = None

class DlDataDeliveryStatus(str, Enum):
    BUFFERED = "BUFFERED"
    TRANSMITTED = "TRANSMITTED"
    DISCARDED = "DISCARDED"

class SatelliteBackhaulCategory(str, Enum):
    GEO = "GEO"
    MEO = "MEO"
    LEO = "LEO"
    OTHER_SAT = "OTHER_SAT"
    NON_SATELLITE = "NON_SATELLITE"

class PcfUeCallbackInfo(BaseModel):
    callbackUri: str
    bindingInfo: str | None = None

class NwdafEvent(str, Enum):
    SLICE_LOAD_LEVEL = "SLICE_LOAD_LEVEL"
    NETWORK_PERFORMANCE = "NETWORK_PERFORMANCE"
    NF_LOAD = "NF_LOAD"
    SERVICE_EXPERIENCE = "SERVICE_EXPERIENCE"
    UE_MOBILITY = "UE_MOBILITY"
    UE_COMMUNICATION = "UE_COMMUNICATION"
    QOS_SUSTAINABILITY = "QOS_SUSTAINABILITY"
    ABNORMAL_BEHAVIOUR = "ABNORMAL_BEHAVIOUR"
    USER_DATA_CONGESTION = "USER_DATA_CONGESTION"
    NSI_LOAD_LEVEL = "NSI_LOAD_LEVEL"
    DN_PERFORMANCE = "DN_PERFORMANCE"
    DISPERSION = "DISPERSION"
    RED_TRANS_EXP = "RED_TRANS_EXP"
    WLAN_PERFORMANCE = "WLAN_PERFORMANCE"
    SM_CONGESTION = "SM_CONGESTION"

class Snssai(BaseModel):
    sst: int
    sd: str | None = None

class NwdafData(BaseModel):
    nwdafInstanceId: str
    nwdafEvents: list[NwdafEvent]

class AccNetChargingAddress(BaseModel):
    anChargIpv4Addr: str | None = None
    anChargIpv6Addr: str | None = None

class PduSessionType(str, Enum):
    IPV4 = "IPV4"
    IPV6 = "IPV6"
    IPV4V6 = "IPV4V6"
    UNSTRUCTURED = "UNSTRUCTURED"
    ETHERNET = "ETHERNET"

class DnnSelectionMode(str, Enum):
    VERIFIED = "VERIFIED"
    UE_DNN_NOT_VERIFIED = "UE_DNN_NOT_VERIFIED"
    NW_DNN_NOT_VERIFIED = "NW_DNN_NOT_VERIFIED"

class ServerAddressingInfo(BaseModel):
    ipv4Addresses: list[str] | None = None
    ipv6Addresses: list[str] | None = None
    fqdnList: list[str] | None = None

class SmPolicyContextData(BaseModel):
    accNetChId: AccNetChId | None = None
    chargEntityAddr: list[AccNetChargingAddress] | None = None
    gpsi: str | None = None
    supi: str | None = None
    invalidSupi: bool | None = None
    interGrpIds: list[str] | None = None
    pduSessionId: int | None = None
    pduSessionType: PduSessionType | None = None
    chargingcharacteristics: str | None = None
    dnn: str | None = None
    dnnSelMode: DnnSelectionMode | None = None
    notificationUri: str | None = None
    accessType: AccessType | None = None
    ratType: RatType | None = None
    addAccessInfo: AdditionalAccessInfo | None = None
    servingNetwork: PlmnIdNid | None = None
    userLocationInfo: UserLocation | None = None
    ueTimeZone: str | None = None
    pei: str | None = None
    ipv4Address: str | None = None
    ipv6AddressPrefix: str | None = None
    ipDomain: str | None = None
    subsSessAmbr: Ambr | None = None
    authProfIndex: str | None = None
    subsDefQos: SubscribedDefaultQos | None = None
    vplmnQos: VplmnQos | None = None
    numOfPackFilter: int | None = None
    online: bool | None = None
    offline: bool | None = None
    var3gppPsDataOffStatus: bool | None = Field(None, alias="3gppPsDataOffStatus")
    refQosIndication: bool | None = None
    traceReq: TraceData | None = None
    sliceInfo: Snssai | None = None
    qosFlowUsage: QosFlowUsage | None = None
    servNfId: ServingNfIdentity | None = None
    suppFeat: str | None = None
    smfId: str | None = None
    recoveryTime: datetime | None = None
    maPduInd: MaPduIndication | None = None
    atsssCapab: PcfSmPolicyControlAtsssCapability | None = None
    ipv4FrameRouteList: list[str] | None = None
    ipv6FrameRouteList: list[str] | None = None
    satBackhaulCategory: SatelliteBackhaulCategory | None = None
    pcfUeInfo: PcfUeCallbackInfo | None = None
    pvsInfo: list[ServerAddressingInfo] | None = None
    onboardInd: bool | None = None
    nwdafDatas: list[NwdafData] | None = None
