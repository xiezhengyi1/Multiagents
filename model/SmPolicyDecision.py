from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from .UserLocation import AccessType, Ambr
from .Arp import Arp
from .RatType import RatType
from .PolicyTrigger import PolicyTrigger
# Enums

class FlowDirection(str, Enum):
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    UNSPECIFIED = "UNSPECIFIED"

class AfSigProtocol(str, Enum):
    NO_INFORMATION = "NO_INFORMATION"
    SIP = "SIP"

class NotificationControlIndication(str, Enum):
    DDN_FAILURE = "DDN_FAILURE"
    DDD_STATUS = "DDD_STATUS"

class DlDataDeliveryStatus(str, Enum):
    BUFFERED = "BUFFERED"
    TRANSMITTED = "TRANSMITTED"
    DISCARDED = "DISCARDED"

class MeteringMethod(str, Enum):
    DURATION = "DURATION"
    VOLUME = "VOLUME"
    DURATION_VOLUME = "DURATION_VOLUME"
    EVENT = "EVENT"

class ReportingLevel(str, Enum):
    SER_ID_LEVEL = "SER_ID_LEVEL"
    RAT_GR_LEVEL = "RAT_GR_LEVEL"
    SPON_CON_LEVEL = "SPON_CON_LEVEL"

class FlowStatus(str, Enum):
    ENABLED_UPLINK = "ENABLED-UPLINK"
    ENABLED_DOWNLINK = "ENABLED-DOWNLINK"
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    REMOVED = "REMOVED"

class RedirectAddressType(str, Enum):
    IPV4_ADDR = "IPV4_ADDR"
    IPV6_ADDR = "IPV6_ADDR"
    URL = "URL"
    SIP_URI = "SIP_URI"

class DnaiChangeType(str, Enum):
    EARLY = "EARLY"
    EARLY_LATE = "EARLY_LATE"
    LATE = "LATE"

class SteeringFunctionality(str, Enum):
    MPTCP = "MPTCP"
    ATSSS_LL = "ATSSS_LL"

class SteerModeValue(str, Enum):
    ACTIVE_STANDBY = "ACTIVE_STANDBY"
    LOAD_BALANCING = "LOAD_BALANCING"
    SMALLEST_DELAY = "SMALLEST_DELAY"
    PRIORITY_BASED = "PRIORITY_BASED"

class SteerModeIndicator(str, Enum):
    AUTO_LOAD_BALANCE = "AUTO_LOAD_BALANCE"
    UE_ASSISTANCE = "UE_ASSISTANCE"

class MulticastAccessControl(str, Enum):
    ALLOWED = "ALLOWED"
    NOT_ALLOWED = "NOT_ALLOWED"

class QosResourceType(str, Enum):
    NON_GBR = "NON_GBR"
    NON_CRITICAL_GBR = "NON_CRITICAL_GBR"
    CRITICAL_GBR = "CRITICAL_GBR"

class RequestedQosMonitoringParameter(str, Enum):
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    ROUND_TRIP = "ROUND_TRIP"

class ReportingFrequency(str, Enum):
    EVENT_TRIGGERED = "EVENT_TRIGGERED"
    PERIODIC = "PERIODIC"

class RequestedRuleDataType(str, Enum):
    CH_ID = "CH_ID"
    MS_TIME_ZONE = "MS_TIME_ZONE"
    USER_LOC_INFO = "USER_LOC_INFO"
    RES_RELEASE = "RES_RELEASE"
    SUCC_RES_ALLO = "SUCC_RES_ALLO"
    EPS_FALLBACK = "EPS_FALLBACK"

class PresenceState(str, Enum):
    IN_AREA = "IN_AREA"
    OUT_OF_AREA = "OUT_OF_AREA"
    UNKNOWN = "UNKNOWN"
    INACTIVE = "INACTIVE"

class QosFlowUsage(str, Enum):
    GENERAL = "GENERAL"
    IMS_SIG = "IMS_SIG"

class SmPolicyAssociationReleaseCause(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    UE_SUBSCRIPTION = "UE_SUBSCRIPTION"
    INSUFFICIENT_RES = "INSUFFICIENT_RES"
    VALIDATION_CONDITION_NOT_MET = "VALIDATION_CONDITION_NOT_MET"
    REACTIVATION_REQUESTED = "REACTIVATION_REQUESTED"

class AuthorizedDefaultQos(BaseModel):
    var5qi: Optional[int] = Field(None, alias="5qi")
    arp: Optional[Arp] = None
    priorityLevel: Optional[int] = None
    averWindow: Optional[int] = None
    maxDataBurstVol: Optional[int] = None
    maxbrUl: Optional[str] = None
    maxbrDl: Optional[str] = None
    gbrUl: Optional[str] = None
    gbrDl: Optional[str] = None
    extMaxDataBurstVol: Optional[int] = None

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

class TscaiInputContainer(BaseModel):
    periodicity: Optional[int] = None
    burstArrivalTime: Optional[datetime] = None
    surTimeInNumMsg: Optional[int] = None
    surTimeInTime: Optional[int] = None

class QosData(BaseModel):
    qosId: str
    var5qi: Optional[int] = Field(None, alias="5qi")
    maxbrUl: Optional[str] = None
    maxbrDl: Optional[str] = None
    gbrUl: Optional[str] = None
    gbrDl: Optional[str] = None
    arp: Optional[Arp] = None
    qnc: Optional[bool] = None
    priorityLevel: Optional[int] = None
    averWindow: Optional[int] = None
    maxDataBurstVol: Optional[int] = None
    reflectiveQos: Optional[bool] = None
    sharingKeyDl: Optional[str] = None
    sharingKeyUl: Optional[str] = None
    maxPacketLossRateDl: Optional[int] = None
    maxPacketLossRateUl: Optional[int] = None
    defQosFlowIndication: Optional[bool] = None
    extMaxDataBurstVol: Optional[int] = None
    packetDelayBudget: Optional[int] = None
    packetErrorRate: Optional[str] = None

class DownlinkDataNotificationControl(BaseModel):
    notifCtrlInds: Optional[List[NotificationControlIndication]] = None
    typesOfNotif: Optional[List[DlDataDeliveryStatus]] = None

class SessionRule(BaseModel):
    authSessAmbr: Optional[Ambr] = None
    authDefQos: Optional[AuthorizedDefaultQos] = None
    sessRuleId: str
    refUmData: Optional[str] = None
    refUmN3gData: Optional[str] = None
    refCondData: Optional[str] = None

class PCCRule(BaseModel):
    flowInfos: Optional[List[FlowInformation]] = None
    appId: Optional[str] = None
    appDescriptor: Optional[str] = None
    contVer: Optional[int] = None
    pccRuleId: str
    precedence: Optional[int] = None
    afSigProtocol: Optional[AfSigProtocol] = None
    appReloc: Optional[bool] = None
    easRedisInd: Optional[bool] = None
    refQosData: Optional[List[str]] = None
    refAltQosParams: Optional[List[str]] = None
    refTcData: Optional[List[str]] = None
    refChgData: Optional[List[str]] = None
    refChgN3gData: Optional[List[str]] = None
    refUmData: Optional[List[str]] = None
    refUmN3gData: Optional[List[str]] = None
    refCondData: Optional[str] = None
    refQosMon: Optional[List[str]] = None
    addrPreserInd: Optional[bool] = None
    tscaiInputDl: Optional[List[TscaiInputContainer]] = None
    tscaiInputUl: Optional[List[TscaiInputContainer]] = None
    tscaiTimeDom: Optional[int] = None
    ddNotifCtrl: Optional[DownlinkDataNotificationControl] = None
    ddNotifCtrl2: Optional[DownlinkDataNotificationControl] = None
    disUeNotif: Optional[bool] = None
    packFiltAllPrec: Optional[int] = None

class ChargingData(BaseModel):
    chgId: str
    meteringMethod: Optional[MeteringMethod] = None
    offline: Optional[bool] = None
    online: Optional[bool] = None
    sdfHandl: Optional[bool] = None
    ratingGroup: Optional[int] = None
    reportingLevel: Optional[ReportingLevel] = None
    serviceId: Optional[int] = None
    sponsorId: Optional[str] = None
    appSvcProvId: Optional[str] = None
    afChargingIdentifier: Optional[int] = None
    afChargId: Optional[str] = None

class ChargingInformation(BaseModel):
    primaryChfAddress: str
    secondaryChfAddress: Optional[str] = None
    primaryChfSetId: Optional[str] = None
    primaryChfInstanceId: Optional[str] = None
    secondaryChfSetId: Optional[str] = None
    secondaryChfInstanceId: Optional[str] = None

class RedirectInformation(BaseModel):
    redirectEnabled: Optional[bool] = None
    redirectAddressType: Optional[RedirectAddressType] = None
    redirectServerAddress: Optional[str] = None

class RouteInformation(BaseModel):
    ipv4Addr: Optional[str] = None
    ipv6Addr: Optional[str] = None
    portNumber: Optional[int] = None

class RouteToLocation(BaseModel):
    dnai: str
    routeInfo: Optional[RouteInformation] = None
    routeProfId: Optional[str] = None

class EasServerAddress(BaseModel):
    ip: str
    port: int

class EasIpReplacementInfo(BaseModel):
    source: Optional[EasServerAddress] = None
    target: Optional[EasServerAddress] = None

class UpPathChgEvent(BaseModel):
    notificationUri: str
    notifCorreId: str
    dnaiChgType: DnaiChangeType
    afAckInd: Optional[bool] = None

class ThresholdValue(BaseModel):
    rttThres: Optional[int] = None
    plrThres: Optional[int] = None

class SteeringMode(BaseModel):
    steerModeValue: SteerModeValue
    active: Optional[AccessType] = None
    standby: Optional[AccessType] = None
    var3gLoad: Optional[int] = Field(None, alias="3gLoad")
    prioAcc: Optional[AccessType] = None
    thresValue: Optional[ThresholdValue] = None
    steerModeInd: Optional[SteerModeIndicator] = None

class TrafficControlData(BaseModel):
    tcId: str
    flowStatus: Optional[FlowStatus] = None
    redirectInfo: Optional[RedirectInformation] = None
    addRedirectInfo: Optional[List[RedirectInformation]] = None
    muteNotif: Optional[bool] = None
    trafficSteeringPolIdDl: Optional[str] = None
    trafficSteeringPolIdUl: Optional[str] = None
    routeToLocs: Optional[List[RouteToLocation]] = None
    maxAllowedUpLat: Optional[int] = None
    easIpReplaceInfos: Optional[List[EasIpReplacementInfo]] = None
    traffCorreInd: Optional[bool] = None
    simConnInd: Optional[bool] = None
    simConnTerm: Optional[int] = None
    upPathChgEvent: Optional[UpPathChgEvent] = None
    steerFun: Optional[SteeringFunctionality] = None
    steerModeDl: Optional[SteeringMode] = None
    steerModeUl: Optional[SteeringMode] = None
    mulAccCtrl: Optional[MulticastAccessControl] = None

class UsageMonitoringData(BaseModel):
    umId: str
    volumeThreshold: Optional[int] = None
    volumeThresholdUplink: Optional[int] = None
    volumeThresholdDownlink: Optional[int] = None
    timeThreshold: Optional[int] = None
    monitoringTime: Optional[datetime] = None
    nextVolThreshold: Optional[int] = None
    nextVolThresholdUplink: Optional[int] = None
    nextVolThresholdDownlink: Optional[int] = None
    nextTimeThreshold: Optional[int] = None
    inactivityTime: Optional[int] = None
    exUsagePccRuleIds: Optional[List[str]] = None

class QosCharacteristics(BaseModel):
    var5qi: int = Field(..., alias="5qi")
    resourceType: QosResourceType
    priorityLevel: int
    packetDelayBudget: int
    packetErrorRate: str
    averagingWindow: Optional[int] = None
    maxDataBurstVol: Optional[int] = None
    extMaxDataBurstVol: Optional[int] = None

class QosMonitoringData(BaseModel):
    qmId: str
    reqQosMonParams: List[RequestedQosMonitoringParameter]
    repFreqs: List[ReportingFrequency]
    repThreshDl: Optional[int] = None
    repThreshUl: Optional[int] = None
    repThreshRp: Optional[int] = None
    waitTime: Optional[int] = None
    repPeriod: Optional[int] = None
    notifyUri: Optional[str] = None
    notifyCorreId: Optional[str] = None
    directNotifInd: Optional[bool] = None

class ConditionData(BaseModel):
    condId: str
    activationTime: Optional[datetime] = None
    deactivationTime: Optional[datetime] = None
    accessType: Optional[List[AccessType]] = None
    ratType: Optional[List[RatType]] = None

class RequestedRuleData(BaseModel):
    refPccRuleIds: List[str]
    reqData: List[RequestedRuleDataType]

class RequestedUsageData(BaseModel):
    refUmIds: Optional[List[str]] = None
    allUmIds: Optional[bool] = None

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class GNbId(BaseModel):
    bitLength: int
    gNBValue: str

class Tai(BaseModel):
    plmnId: PlmnId
    tac: str
    nid: Optional[str] = None

class Ecgi(BaseModel):
    plmnId: PlmnId
    eutraCellId: str
    nid: Optional[str] = None

class Ncgi(BaseModel):
    plmnId: PlmnId
    nrCellId: str
    nid: Optional[str] = None

class GlobalRanNodeId(BaseModel):
    plmnId: PlmnId
    n3IwfId: Optional[str] = None
    gNbId: Optional[GNbId] = None
    ngeNbId: Optional[str] = None
    wagfId: Optional[str] = None
    tngfId: Optional[str] = None
    twifId: Optional[str] = None
    nid: Optional[str] = None
    eNbId: Optional[str] = None

class PresenceInfoRm(BaseModel):
    praId: Optional[str] = None
    additionalPraId: Optional[str] = None
    presenceState: Optional[PresenceState] = None
    trackingAreaList: Optional[List[Tai]] = None
    ecgiList: Optional[List[Ecgi]] = None
    ncgiList: Optional[List[Ncgi]] = None
    globalRanNodeIdList: Optional[List[GlobalRanNodeId]] = None
    globaleNbIdList: Optional[List[GlobalRanNodeId]] = None

class BridgeManagementContainer(BaseModel):
    bridgeManCont: str

class PortManagementContainer(BaseModel):
    portManCont: str
    portNum: int

class SmPolicyDecision(BaseModel):
    sessRules: Optional[Dict[str, SessionRule]] = None
    pccRules: Optional[Dict[str, PCCRule]] = None
    pcscfRestIndication: Optional[bool] = None
    qosDecs: Optional[Dict[str, QosData]] = None
    chgDecs: Optional[Dict[str, ChargingData]] = None
    chargingInfo: Optional[ChargingInformation] = None
    traffContDecs: Optional[Dict[str, TrafficControlData]] = None
    umDecs: Optional[Dict[str, UsageMonitoringData]] = None
    qosChars: Optional[Dict[str, QosCharacteristics]] = None
    qosMonDecs: Optional[Dict[str, QosMonitoringData]] = None
    reflectiveQoSTimer: Optional[int] = None
    conds: Optional[Dict[str, ConditionData]] = None
    revalidationTime: Optional[datetime] = None
    offline: Optional[bool] = None
    online: Optional[bool] = None
    offlineChOnly: Optional[bool] = None
    policyCtrlReqTriggers: Optional[List[PolicyTrigger]] = None
    lastReqRuleData: Optional[List[RequestedRuleData]] = None
    lastReqUsageData: Optional[RequestedUsageData] = None
    praInfos: Optional[Dict[str, PresenceInfoRm]] = None
    ipv4Index: Optional[int] = None
    ipv6Index: Optional[int] = None
    qosFlowUsage: Optional[QosFlowUsage] = None
    relCause: Optional[SmPolicyAssociationReleaseCause] = None
    suppFeat: Optional[str] = None
    tsnBridgeManCont: Optional[BridgeManagementContainer] = None
    tsnPortManContDstt: Optional[PortManagementContainer] = None
    tsnPortManContNwtts: Optional[List[PortManagementContainer]] = None
    redSessIndication: Optional[bool] = None
