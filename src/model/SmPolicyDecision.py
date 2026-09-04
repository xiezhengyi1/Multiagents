from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .Arp import Arp
from .PolicyTrigger import PolicyTrigger
from .RatType import RatType
from .UserLocation import AccessType, Ambr

# Enums


class OpenAPIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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


class AuthorizedDefaultQos(OpenAPIBaseModel):
    var5qi: int | None = Field(None, alias="5qi")
    arp: Arp | None = None
    priorityLevel: int | None = None
    averWindow: int | None = None
    maxDataBurstVol: int | None = None
    maxbrUl: str | None = None
    maxbrDl: str | None = None
    gbrUl: str | None = None
    gbrDl: str | None = None
    extMaxDataBurstVol: int | None = None


class EthFlowDescription(OpenAPIBaseModel):
    destMacAddr: str | None = None
    ethType: str
    fDesc: str | None = None
    fDir: FlowDirection | None = None
    sourceMacAddr: str | None = None
    vlanTags: list[str] | None = None
    srcMacAddrEnd: str | None = None
    destMacAddrEnd: str | None = None


class FlowInformation(OpenAPIBaseModel):
    flowDescription: str | None = None
    ethFlowDescription: EthFlowDescription | None = None
    packFiltId: str | None = None
    packetFilterUsage: bool | None = None
    tosTrafficClass: str | None = None
    spi: str | None = None
    flowLabel: str | None = None
    flowDirection: FlowDirection | None = None


class TscaiInputContainer(OpenAPIBaseModel):
    periodicity: int | None = None
    burstArrivalTime: datetime | None = None
    surTimeInNumMsg: int | None = None
    surTimeInTime: int | None = None


class QosData(OpenAPIBaseModel):
    qosId: str | None = None
    var5qi: int | None = Field(None, alias="5qi")
    maxbrUl: str | None = None
    maxbrDl: str | None = None
    gbrUl: str | None = None
    gbrDl: str | None = None
    arp: Arp | None = None
    qnc: bool | None = None
    priorityLevel: int | None = None
    averWindow: int | None = None
    maxDataBurstVol: int | None = None
    reflectiveQos: bool | None = None
    sharingKeyDl: str | None = None
    sharingKeyUl: str | None = None
    maxPacketLossRateDl: int | None = None
    maxPacketLossRateUl: int | None = None
    defQosFlowIndication: bool | None = None
    extMaxDataBurstVol: int | None = None
    packetDelayBudget: int | None = None
    packetErrorRate: str | None = None


class DownlinkDataNotificationControl(OpenAPIBaseModel):
    notifCtrlInds: list[NotificationControlIndication] | None = None
    typesOfNotif: list[DlDataDeliveryStatus] | None = None


class SessionRule(OpenAPIBaseModel):
    authSessAmbr: Ambr | None = None
    authDefQos: AuthorizedDefaultQos | None = None
    sessRuleId: str
    refUmData: str | None = None
    refUmN3gData: str | None = None
    refCondData: str | None = None


class PCCRule(OpenAPIBaseModel):
    flowInfos: list[FlowInformation] | None = None
    appId: str | None = None
    appDescriptor: str | None = None
    contVer: int | None = None
    pccRuleId: str
    precedence: int | None = None
    afSigProtocol: AfSigProtocol | None = None
    appReloc: bool | None = None
    easRedisInd: bool | None = None
    refQosData: list[str] | None = None
    refAltQosParams: list[str] | None = None
    refTcData: list[str] | None = None
    refChgData: list[str] | None = None
    refChgN3gData: list[str] | None = None
    refUmData: list[str] | None = None
    refUmN3gData: list[str] | None = None
    refCondData: str | None = None
    refQosMon: list[str] | None = None
    addrPreserInd: bool | None = None
    tscaiInputDl: list[TscaiInputContainer] | None = None
    tscaiInputUl: list[TscaiInputContainer] | None = None
    tscaiTimeDom: int | None = None
    ddNotifCtrl: DownlinkDataNotificationControl | None = None
    ddNotifCtrl2: DownlinkDataNotificationControl | None = None
    disUeNotif: bool | None = None
    packFiltAllPrec: int | None = None


class ChargingData(OpenAPIBaseModel):
    chgId: str
    meteringMethod: MeteringMethod | None = None
    offline: bool | None = None
    online: bool | None = None
    sdfHandl: bool | None = None
    ratingGroup: int | None = None
    reportingLevel: ReportingLevel | None = None
    serviceId: int | None = None
    sponsorId: str | None = None
    appSvcProvId: str | None = None
    afChargingIdentifier: int | None = None
    afChargId: str | None = None


class ChargingInformation(OpenAPIBaseModel):
    primaryChfAddress: str
    secondaryChfAddress: str | None = None
    primaryChfSetId: str | None = None
    primaryChfInstanceId: str | None = None
    secondaryChfSetId: str | None = None
    secondaryChfInstanceId: str | None = None


class RedirectInformation(OpenAPIBaseModel):
    redirectEnabled: bool | None = None
    redirectAddressType: RedirectAddressType | None = None
    redirectServerAddress: str | None = None


class RouteInformation(OpenAPIBaseModel):
    ipv4Addr: str | None = None
    ipv6Addr: str | None = None
    portNumber: int | None = None


class RouteToLocation(OpenAPIBaseModel):
    dnai: str
    routeInfo: RouteInformation | None = None
    routeProfId: str | None = None


class EasServerAddress(OpenAPIBaseModel):
    ip: str
    port: int


class EasIpReplacementInfo(OpenAPIBaseModel):
    source: EasServerAddress | None = None
    target: EasServerAddress | None = None


class UpPathChgEvent(OpenAPIBaseModel):
    notificationUri: str
    notifCorreId: str
    dnaiChgType: DnaiChangeType
    afAckInd: bool | None = None


class ThresholdValue(OpenAPIBaseModel):
    rttThres: int | None = None
    plrThres: int | None = None


class SteeringMode(OpenAPIBaseModel):
    steerModeValue: SteerModeValue
    active: AccessType | None = None
    standby: AccessType | None = None
    var3gLoad: int | None = Field(None, alias="3gLoad")
    prioAcc: AccessType | None = None
    thresValue: ThresholdValue | None = None
    steerModeInd: SteerModeIndicator | None = None


class TrafficControlData(OpenAPIBaseModel):
    tcId: str
    flowStatus: FlowStatus | None = None
    redirectInfo: RedirectInformation | None = None
    addRedirectInfo: list[RedirectInformation] | None = None
    muteNotif: bool | None = None
    trafficSteeringPolIdDl: str | None = None
    trafficSteeringPolIdUl: str | None = None
    routeToLocs: list[RouteToLocation] | None = None
    maxAllowedUpLat: int | None = None
    easIpReplaceInfos: list[EasIpReplacementInfo] | None = None
    traffCorreInd: bool | None = None
    simConnInd: bool | None = None
    simConnTerm: int | None = None
    upPathChgEvent: UpPathChgEvent | None = None
    steerFun: SteeringFunctionality | None = None
    steerModeDl: SteeringMode | None = None
    steerModeUl: SteeringMode | None = None
    mulAccCtrl: MulticastAccessControl | None = None


class UsageMonitoringData(OpenAPIBaseModel):
    umId: str
    volumeThreshold: int | None = None
    volumeThresholdUplink: int | None = None
    volumeThresholdDownlink: int | None = None
    timeThreshold: int | None = None
    monitoringTime: datetime | None = None
    nextVolThreshold: int | None = None
    nextVolThresholdUplink: int | None = None
    nextVolThresholdDownlink: int | None = None
    nextTimeThreshold: int | None = None
    inactivityTime: int | None = None
    exUsagePccRuleIds: list[str] | None = None


class QosCharacteristics(OpenAPIBaseModel):
    var5qi: int = Field(..., alias="5qi")
    resourceType: QosResourceType
    priorityLevel: int
    packetDelayBudget: int
    packetErrorRate: str
    averagingWindow: int | None = None
    maxDataBurstVol: int | None = None
    extMaxDataBurstVol: int | None = None


class QosMonitoringData(OpenAPIBaseModel):
    qmId: str
    reqQosMonParams: list[RequestedQosMonitoringParameter]
    repFreqs: list[ReportingFrequency]
    repThreshDl: int | None = None
    repThreshUl: int | None = None
    repThreshRp: int | None = None
    waitTime: int | None = None
    repPeriod: int | None = None
    notifyUri: str | None = None
    notifyCorreId: str | None = None
    directNotifInd: bool | None = None


class ConditionData(OpenAPIBaseModel):
    condId: str
    activationTime: datetime | None = None
    deactivationTime: datetime | None = None
    accessType: list[AccessType] | None = None
    ratType: list[RatType] | None = None


class RequestedRuleData(OpenAPIBaseModel):
    refPccRuleIds: list[str]
    reqData: list[RequestedRuleDataType]


class RequestedUsageData(OpenAPIBaseModel):
    refUmIds: list[str] | None = None
    allUmIds: bool | None = None


class PlmnId(OpenAPIBaseModel):
    mcc: str
    mnc: str


class GNbId(OpenAPIBaseModel):
    bitLength: int
    gNBValue: str


class Tai(OpenAPIBaseModel):
    plmnId: PlmnId
    tac: str
    nid: str | None = None


class Ecgi(OpenAPIBaseModel):
    plmnId: PlmnId
    eutraCellId: str
    nid: str | None = None


class Ncgi(OpenAPIBaseModel):
    plmnId: PlmnId
    nrCellId: str
    nid: str | None = None


class GlobalRanNodeId(OpenAPIBaseModel):
    plmnId: PlmnId
    n3IwfId: str | None = None
    gNbId: GNbId | None = None
    ngeNbId: str | None = None
    wagfId: str | None = None
    tngfId: str | None = None
    twifId: str | None = None
    nid: str | None = None
    eNbId: str | None = None


class PresenceInfoRm(OpenAPIBaseModel):
    praId: str | None = None
    additionalPraId: str | None = None
    presenceState: PresenceState | None = None
    trackingAreaList: list[Tai] | None = None
    ecgiList: list[Ecgi] | None = None
    ncgiList: list[Ncgi] | None = None
    globalRanNodeIdList: list[GlobalRanNodeId] | None = None
    globaleNbIdList: list[GlobalRanNodeId] | None = None


class BridgeManagementContainer(OpenAPIBaseModel):
    bridgeManCont: str


class PortManagementContainer(OpenAPIBaseModel):
    portManCont: str
    portNum: int


class SmPolicyDecision(OpenAPIBaseModel):
    sessRules: dict[str, SessionRule] | None = None
    pccRules: dict[str, PCCRule] | None = None
    pcscfRestIndication: bool | None = None
    qosDecs: dict[str, QosData] | None = None
    chgDecs: dict[str, ChargingData] | None = None
    chargingInfo: ChargingInformation | None = None
    traffContDecs: dict[str, TrafficControlData] | None = None
    umDecs: dict[str, UsageMonitoringData] | None = None
    qosChars: dict[str, QosCharacteristics] | None = None
    qosMonDecs: dict[str, QosMonitoringData] | None = None
    reflectiveQoSTimer: int | None = None
    conds: dict[str, ConditionData] | None = None
    revalidationTime: datetime | None = None
    offline: bool | None = None
    online: bool | None = None
    offlineChOnly: bool | None = None
    policyCtrlReqTriggers: list[PolicyTrigger] | None = None
    lastReqRuleData: list[RequestedRuleData] | None = None
    lastReqUsageData: RequestedUsageData | None = None
    praInfos: dict[str, PresenceInfoRm] | None = None
    ipv4Index: int | None = None
    ipv6Index: int | None = None
    qosFlowUsage: QosFlowUsage | None = None
    relCause: SmPolicyAssociationReleaseCause | None = None
    suppFeat: str | None = None
    tsnBridgeManCont: BridgeManagementContainer | None = None
    tsnPortManContDstt: PortManagementContainer | None = None
    tsnPortManContNwtts: list[PortManagementContainer] | None = None
    redSessIndication: bool | None = None
