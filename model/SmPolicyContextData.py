from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from .UserLocation import Tai, Ecgi, Ncgi, GlobalRanNodeId, AccessType, UserLocation, Ambr
from .RatType import RatType
from .Arp import Arp
# --- Sm Policy Context Data Definitions ---

class SubscribedDefaultQos(BaseModel):
    var5qi: int = Field(..., alias="5qi")
    arp: Optional[Arp] = None
    priorityLevel: Optional[int] = None

class VplmnQos(BaseModel):
    var5qi: Optional[int] = Field(None, alias="5qi")
    arp: Optional[Arp] = None
    sessionAmbr: Optional[Ambr] = None
    maxFbrDl: Optional[str] = None
    maxFbrUl: Optional[str] = None
    guaFbrDl: Optional[str] = None
    guaFbrUl: Optional[str] = None

class AccuUsageReports(BaseModel):
    refUmIds: str
    volUsage: Optional[int] = None
    volUsageUplink: Optional[int] = None
    volUsageDownlink: Optional[int] = None
    timeUsage: Optional[int] = None
    nextVolUsage: Optional[int] = None
    nextVolUsageUplink: Optional[int] = None
    nextVolUsageDownlink: Optional[int] = None
    nextTimeUsage: Optional[int] = None

class PresenceState(str, Enum):
    IN_AREA = "IN_AREA"
    OUT_OF_AREA = "OUT_OF_AREA"
    UNKNOWN = "UNKNOWN"
    INACTIVE = "INACTIVE"

class RequestQos(BaseModel):
    var5qi: int
    gbrUl: Optional[str] = None
    gbrDl: Optional[str] = None

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
    accNetChaIdValue: Optional[int] = None
    accNetChargIdString: Optional[str] = None
    refPccRuleIds: Optional[List[str]] = None
    sessionChScope: Optional[bool] = None

class AccessType(str, Enum):
    _3GPP_ACCESS = "3GPP_ACCESS"
    NON_3GPP_ACCESS = "NON_3GPP_ACCESS"

class AdditionalAccessInfo(BaseModel):
    accessType: Optional[List[AccessType]] = None
    ratType: Optional[RatType] = None

class PlmnIdNid(BaseModel):
    mcc: str
    mnc: str
    nid: Optional[str] = None

class EthFlowDescription(BaseModel):
    destMacAddr: Optional[str] = None
    ethType: str
    fDesc: Optional[str] = None
    fDir: Optional[FlowDirection] = None
    sourceMacAddr: Optional[str] = None
    vlanTags: Optional[List[str]] = None
    srcMacAddrEnd: Optional[str] = None
    destMacAddrEnd: Optional[str] = None

class FlowInformation(BaseModel):
    flowDescription: Optional[str] = None
    ethFlowDescription: Optional[EthFlowDescription] = None
    packFiltId: Optional[str] = None
    packetFilterUsage: Optional[bool] = None
    tosTrafficClass: Optional[str] = None
    spi: Optional[str] = None
    flowLabel: Optional[str] = None
    flowDirection: Optional[FlowDirection] = None

class AppDetectionInfo(BaseModel):
    appId: str
    instanceId: Optional[str] = None
    sdfDescriptions: Optional[List[FlowInformation]] = None

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
    ngApCause: Optional[NgApCause] = None
    var5gMmCause: Optional[int] = Field(None, alias="5gMmCause")
    var5gSmCause: Optional[int] = Field(None, alias="5gSmCause")
    epsCause: Optional[str] = None

class RuleReport(BaseModel):
    pccRuleIds: List[str]
    ruleStatus: List[RuleStatus]
    contVers: Optional[List[int]] = None
    failureCode: Optional[List[PcfSmPolicyControlFailureCode]] = None
    finUnitAct: Optional[List[FinalUnitAction]] = None
    ranNasRelCauses: Optional[List[RanNasRelCause]] = None
    altQosParamId: Optional[str] = None

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
    ruleIds: List[str]
    ruleStatus: List[RuleStatus]
    sessRuleFailureCode: Optional[List[SessionRuleFailureCode]] = None

class QosNotifType(str, Enum):
    GUARANTEED = "GUARANTEED"
    NOT_GUARANTEED = "NOT_GUARANTEED"

class PcfSmPolicyControlQosNotificationControlInfo(BaseModel):
    refPccRuleIds: List[str]
    notifType: List[QosNotifType]
    contVer: Optional[int] = None
    altQosParamId: Optional[str] = None

class PcfSmPolicyControlQosMonitoringReport(BaseModel):
    refPccRuleIds: List[str]
    ulDelays: Optional[List[int]] = None
    dlDelays: Optional[List[int]] = None
    rtDelays: Optional[List[int]] = None
    pdmf: Optional[bool] = None

class PresenceInfo(BaseModel):
    praId: Optional[str] = None
    fDir: Optional[FlowDirection] = None
    presenceState: Optional[PresenceState] = None
    trackingAreaList: Optional[List[Tai]] = None
    ecgiList: Optional[List[Ecgi]] = None
    ncgiList: Optional[List[Ncgi]] = None
    globalRanNodeIdList: Optional[List[GlobalRanNodeId]] = None
    globaleNbIdList: Optional[List[GlobalRanNodeId]] = None

class PacketFilterInfo(BaseModel):
    packFiltId: Optional[str] = None
    packFiltCont: Optional[str] = None
    tosTrafficClass: Optional[str] = None
    spi: Optional[str] = None
    flowLabel: Optional[str] = None
    flowDirection: Optional[FlowDirection] = None

class UeInitiatedResourceRequest(BaseModel):
    pccRuleId: Optional[str] = None
    ruleOp: RuleOperation
    precedence: Optional[int] = None
    packFiltInfo: List[PacketFilterInfo]
    reqQos: Optional[RequestQos] = None

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
    anGwIpv4Addr: Optional[str] = None
    anGwIpv6Addr: Optional[str] = None

class SgsnAddress(BaseModel):
    sgsnIpv4Addr: Optional[str] = None
    sgsnIpv6Addr: Optional[str] = None

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
    neTypeList: Optional[str] = None
    eventList: Optional[str] = None
    collectionEntityIpv4Addr: Optional[str] = None
    collectionEntityIpv6Addr: Optional[str] = None
    interfaceList: Optional[str] = None

class ServingNfIdentity(BaseModel):
    servNfInstId: Optional[str] = None
    guami: Optional[Guami] = None
    anGwAddr: Optional[AnGwAddress] = None
    sgsnAddr: Optional[SgsnAddress] = None

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
    bridgeId: Optional[int] = None
    dsttAddr: Optional[str] = None
    dsttPortNum: Optional[int] = None
    dsttResidTime: Optional[int] = None

class BridgeManagementContainer(BaseModel):
    bridgeManCont: str

class PortManagementContainer(BaseModel):
    portManCont: str
    portNum: int

class IpMulticastAddressInfo(BaseModel):
    srcIpv4Addr: Optional[str] = None
    ipv4MulAddr: Optional[str] = None
    srcIpv6Addr: Optional[str] = None
    ipv6MulAddr: Optional[str] = None

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
    reason: Optional[str] = None

class DddTrafficDescriptor(BaseModel):
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    portNumber: Optional[int] = None
    macAddr: Optional[str] = None

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
    bindingInfo: Optional[str] = None

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
    sd: Optional[str] = None

class NwdafData(BaseModel):
    nwdafInstanceId: str
    nwdafEvents: List[NwdafEvent]

class AccNetChargingAddress(BaseModel):
    anChargIpv4Addr: Optional[str] = None
    anChargIpv6Addr: Optional[str] = None

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
    ipv4Addresses: Optional[List[str]] = None
    ipv6Addresses: Optional[List[str]] = None
    fqdnList: Optional[List[str]] = None

class SmPolicyContextData(BaseModel):
    accNetChId: Optional[AccNetChId] = None
    chargEntityAddr: Optional[List[AccNetChargingAddress]] = None
    gpsi: Optional[str] = None
    supi: Optional[str] = None
    invalidSupi: Optional[bool] = None
    interGrpIds: Optional[List[str]] = None
    pduSessionId: Optional[int] = None
    pduSessionType: Optional[PduSessionType] = None
    chargingcharacteristics: Optional[str] = None
    dnn: Optional[str] = None
    dnnSelMode: Optional[DnnSelectionMode] = None
    notificationUri: Optional[str] = None
    accessType: Optional[AccessType] = None
    ratType: Optional[RatType] = None
    addAccessInfo: Optional[AdditionalAccessInfo] = None
    servingNetwork: Optional[PlmnIdNid] = None
    userLocationInfo: Optional[UserLocation] = None
    ueTimeZone: Optional[str] = None
    pei: Optional[str] = None
    ipv4Address: Optional[str] = None
    ipv6AddressPrefix: Optional[str] = None
    ipDomain: Optional[str] = None
    subsSessAmbr: Optional[Ambr] = None
    authProfIndex: Optional[str] = None
    subsDefQos: Optional[SubscribedDefaultQos] = None
    vplmnQos: Optional[VplmnQos] = None
    numOfPackFilter: Optional[int] = None
    online: Optional[bool] = None
    offline: Optional[bool] = None
    var3gppPsDataOffStatus: Optional[bool] = Field(None, alias="3gppPsDataOffStatus")
    refQosIndication: Optional[bool] = None
    traceReq: Optional[TraceData] = None
    sliceInfo: Optional[Snssai] = None
    qosFlowUsage: Optional[QosFlowUsage] = None
    servNfId: Optional[ServingNfIdentity] = None
    suppFeat: Optional[str] = None
    smfId: Optional[str] = None
    recoveryTime: Optional[datetime] = None
    maPduInd: Optional[MaPduIndication] = None
    atsssCapab: Optional[PcfSmPolicyControlAtsssCapability] = None
    ipv4FrameRouteList: Optional[List[str]] = None
    ipv6FrameRouteList: Optional[List[str]] = None
    satBackhaulCategory: Optional[SatelliteBackhaulCategory] = None
    pcfUeInfo: Optional[PcfUeCallbackInfo] = None
    pvsInfo: Optional[List[ServerAddressingInfo]] = None
    onboardInd: Optional[bool] = None
    nwdafDatas: Optional[List[NwdafData]] = None
