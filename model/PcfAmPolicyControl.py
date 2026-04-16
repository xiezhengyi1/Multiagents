from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OpenAPIBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid')


class PcfAmPolicyControlPolicyAssociation(OpenAPIBaseModel):
    """Represents an individual AM Policy Association resource."""
    request: Optional[PcfAmPolicyControlPolicyAssociationRequest] = Field(None)
    triggers: Optional[List[PcfAmPolicyControlRequestTrigger]] = Field(None, description="Request Triggers that the PCF subscribes.")
    servAreaRes: Optional[ServiceAreaRestriction] = Field(None)
    wlServAreaRes: Optional[WirelineServiceAreaRestriction] = Field(None)
    rfsp: Optional[int] = Field(None, description="Unsigned integer representing the \\\"Subscriber Profile ID for RAT/Frequency Priority\\\"  as specified in 3GPP TS 36.413.")
    targetRfsp: Optional[int] = Field(None, description="Unsigned integer representing the \\\"Subscriber Profile ID for RAT/Frequency Priority\\\"  as specified in 3GPP TS 36.413.")
    smfSelInfo: Optional[SmfSelectionData] = Field(None)
    ueAmbr: Optional[Ambr] = Field(None)
    ueSliceMbrs: Optional[List[UeSliceMbr]] = Field(None, description="One or more UE-Slice-MBR(s) for S-NSSAI(s) of serving PLMN as part of the AMF Access and Mobility Policy as determined by the PCF.")
    pras: Optional[Dict[str, PresenceInfo]] = Field(None, description="Contains the presence reporting area(s) for which reporting was requested. The praId attribute within the PresenceInfo data type is the key of the map.")
    suppFeat: str = Field(..., description="A string used to indicate the features supported by an API that is used as defined in clause  6.6 in 3GPP TS 29.500. The string shall contain a bitmask indicating supported features in  hexadecimal representation Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\",  \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent the support of 4 features as described in  table 5.2.2-3. The most significant character representing the highest-numbered features shall  appear first in the string, and the character representing features 1 to 4 shall appear last  in the string. The list of features and their numbering (starting with 1) are defined  separately for each API. If the string contains a lower number of characters than there are  defined features for an API, all features that would be represented by characters that are not  present in the string are not supported.")
    pcfUeInfo: Optional[PcfUeCallbackInfo] = Field(None)
    matchPdus: Optional[List[PduSessionInfo]] = Field(None)
    asTimeDisParam: Optional[PcfAmPolicyControlAsTimeDistributionParam] = Field(None)


class PcfAmPolicyControlPolicyAssociationRequest(OpenAPIBaseModel):
    """Information which the NF service consumer provides when requesting the creation of a policy association. The serviveName property corresponds to the serviceName in the main body of the specification."""
    notificationUri: str = Field(..., description="String providing an URI formatted according to RFC 3986.")
    altNotifIpv4Addrs: Optional[List[str]] = Field(None, description="Alternate or backup IPv4 Address(es) where to send Notifications.")
    altNotifIpv6Addrs: Optional[List[str]] = Field(None, description="Alternate or backup IPv6 Address(es) where to send Notifications.")
    altNotifFqdns: Optional[List[str]] = Field(None, description="Alternate or backup FQDN(s) where to send Notifications.")
    supi: str = Field(..., description="String identifying a Supi that shall contain either an IMSI, a network specific identifier, a Global Cable Identifier (GCI) or a Global Line Identifier (GLI) as specified in clause  2.2A of 3GPP TS 23.003. It shall be formatted as follows  - for an IMSI \\\"imsi-<imsi>\\\", where <imsi> shall be formatted according to clause 2.2    of 3GPP TS 23.003 that describes an IMSI.  - for a network specific identifier \\\"nai-<nai>, where <nai> shall be formatted    according to clause 28.7.2 of 3GPP TS 23.003 that describes an NAI.  - for a GCI \\\"gci-<gci>\\\", where <gci> shall be formatted according to clause 28.15.2    of 3GPP TS 23.003.  - for a GLI \\\"gli-<gli>\\\", where <gli> shall be formatted according to clause 28.16.2 of    3GPP TS 23.003.To enable that the value is used as part of an URI, the string shall    only contain characters allowed according to the \\\"lower-with-hyphen\\\" naming convention    defined in 3GPP TS 29.501.")
    gpsi: Optional[str] = Field(None, description="String identifying a Gpsi shall contain either an External Id or an MSISDN.  It shall be formatted as follows -External Identifier= \\\"extid-'extid', where 'extid'  shall be formatted according to clause 19.7.2 of 3GPP TS 23.003 that describes an  External Identifier.")
    accessType: Optional[AccessType] = Field(None)
    accessTypes: Optional[List[AccessType]] = Field(None)
    pei: Optional[str] = Field(None, description="String representing a Permanent Equipment Identifier that may contain - an IMEI or IMEISV, as  specified in clause 6.2 of 3GPP TS 23.003; a MAC address for a 5G-RG or FN-RG via  wireline  access, with an indication that this address cannot be trusted for regulatory purpose if this  address cannot be used as an Equipment Identifier of the FN-RG, as specified in clause 4.7.7  of 3GPP TS23.316. Examples are imei-012345678901234 or imeisv-0123456789012345.")
    userLoc: Optional[UserLocation] = Field(None)
    timeZone: Optional[str] = Field(None, description="String with format \\\"time-numoffset\\\" optionally appended by \\\"daylightSavingTime\\\", where  - \\\"time-numoffset\\\" shall represent the time zone adjusted for daylight saving time and be    encoded as time-numoffset as defined in clause 5.6 of IETF RFC 3339;  - \\\"daylightSavingTime\\\" shall represent the adjustment that has been made and shall be    encoded as \\\"+1\\\" or \\\"+2\\\" for a +1 or +2 hours adjustment.   The example is for 8 hours behind UTC, +1 hour adjustment for Daylight Saving Time.")
    servingPlmn: Optional[PlmnIdNid] = Field(None)
    ratType: Optional[RatType] = Field(None)
    ratTypes: Optional[List[RatType]] = Field(None)
    groupIds: Optional[List[str]] = Field(None)
    servAreaRes: Optional[ServiceAreaRestriction] = Field(None)
    wlServAreaRes: Optional[WirelineServiceAreaRestriction] = Field(None)
    rfsp: Optional[int] = Field(None, description="Unsigned integer representing the \\\"Subscriber Profile ID for RAT/Frequency Priority\\\"  as specified in 3GPP TS 36.413.")
    ueAmbr: Optional[Ambr] = Field(None)
    ueSliceMbrs: Optional[List[UeSliceMbr]] = Field(None, description="The subscribed UE Slice-MBR for each subscribed S-NSSAI of the home PLMN mapping  to a S-NSSAI of the serving PLMN Shall be provided when available.")
    allowedSnssais: Optional[List[Snssai]] = Field(None, description="array of allowed S-NSSAIs for the 3GPP access.")
    targetSnssais: Optional[List[Snssai]] = Field(None, description="array of target S-NSSAIs.")
    mappingSnssais: Optional[List[MappingOfSnssai]] = Field(None, description="mapping of each S-NSSAI of the Allowed NSSAI to the corresponding S-NSSAI of the HPLMN.")
    n3gAllowedSnssais: Optional[List[Snssai]] = Field(None, description="array of allowed S-NSSAIs for the Non-3GPP access.")
    guami: Optional[Guami] = Field(None)
    serviceName: Optional[ServiceName] = Field(None)
    traceReq: Optional[TraceData] = Field(None)
    nwdafDatas: Optional[List[NwdafData]] = Field(None)
    suppFeat: str = Field(..., description="A string used to indicate the features supported by an API that is used as defined in clause  6.6 in 3GPP TS 29.500. The string shall contain a bitmask indicating supported features in  hexadecimal representation Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\",  \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent the support of 4 features as described in  table 5.2.2-3. The most significant character representing the highest-numbered features shall  appear first in the string, and the character representing features 1 to 4 shall appear last  in the string. The list of features and their numbering (starting with 1) are defined  separately for each API. If the string contains a lower number of characters than there are  defined features for an API, all features that would be represented by characters that are not  present in the string are not supported.")


class PcfAmPolicyControlRequestTrigger(str, Enum):
    LOC_CH = "LOC_CH"
    PRA_CH = "PRA_CH"
    SERV_AREA_CH = "SERV_AREA_CH"
    RFSP_CH = "RFSP_CH"
    ALLOWED_NSSAI_CH = "ALLOWED_NSSAI_CH"
    UE_AMBR_CH = "UE_AMBR_CH"
    UE_SLICE_MBR_CH = "UE_SLICE_MBR_CH"
    SMF_SELECT_CH = "SMF_SELECT_CH"
    ACCESS_TYPE_CH = "ACCESS_TYPE_CH"
    NWDAF_DATA_CH = "NWDAF_DATA_CH"
    TARGET_NSSAI = "TARGET_NSSAI"


class ServiceAreaRestriction(OpenAPIBaseModel):
    """Provides information about allowed or not allowed areas."""
    restrictionType: Optional[RestrictionType] = Field(None)
    areas: Optional[List[Area]] = Field(None)
    maxNumOfTAs: Optional[int] = Field(None, description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible.")
    maxNumOfTAsForNotAllowedAreas: Optional[int] = Field(None, description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible.")


class WirelineServiceAreaRestriction(OpenAPIBaseModel):
    """The \"restrictionType\" attribute and the \"areas\" attribute shall be either both present or absent.  The empty array of areas is used when service is allowed/restricted nowhere."""
    restrictionType: Optional[RestrictionType] = Field(None)
    areas: Optional[List[WirelineArea]] = Field(None)


class SmfSelectionData(OpenAPIBaseModel):
    """Represents the SMF Selection information that may be replaced by the PCF."""
    unsuppDnn: Optional[bool] = Field(None)
    candidates: Optional[Dict[str, CandidateForReplacement]] = Field(None, description="Contains the list of DNNs per S-NSSAI that are candidates for replacement. The snssai attribute within the CandidateForReplacement data type is the key of the map.")
    snssai: Optional[Snssai] = Field(None)
    mappingSnssai: Optional[Snssai] = Field(None)
    dnn: Optional[str] = Field(None, description="String representing a Data Network as defined in clause 9A of 3GPP TS 23.003;  it shall contain either a DNN Network Identifier, or a full DNN with both the Network  Identifier and Operator Identifier, as specified in 3GPP TS 23.003 clause 9.1.1 and 9.1.2. It shall be coded as string in which the labels are separated by dots  (e.g. \\\"Label1.Label2.Label3\\\").")


class Ambr(OpenAPIBaseModel):
    """Contains the maximum aggregated uplink and downlink bit rates."""
    uplink: str = Field(..., description="String representing a bit rate; the prefixes follow the standard symbols from The International System of Units, and represent x1000 multipliers, with the exception that prefix \\\"K\\\" is used to represent the standard symbol \\\"k\\\".")
    downlink: str = Field(..., description="String representing a bit rate; the prefixes follow the standard symbols from The International System of Units, and represent x1000 multipliers, with the exception that prefix \\\"K\\\" is used to represent the standard symbol \\\"k\\\".")


class UeSliceMbr(OpenAPIBaseModel):
    """Contains a UE-Slice-MBR and the related information."""
    sliceMbr: Dict[str, SliceMbr] = Field(..., description="Contains the MBR for uplink and the MBR for downlink.")
    servingSnssai: Optional[Snssai] = Field(None)
    mappedHomeSnssai: Optional[Snssai] = Field(None)


class PresenceInfo(OpenAPIBaseModel):
    """If the additionalPraId IE is present, this IE shall state the presence information of the UE for the individual PRA identified by the additionalPraId IE;  If the additionalPraId IE is not present, this IE shall state the presence information of the UE for the PRA identified by the praId IE."""
    praId: Optional[str] = Field(None, description="Represents an identifier of the Presence Reporting Area (see clause 28.10 of 3GPP  TS 23.003.  This IE shall be present  if the Area of Interest subscribed or reported is a Presence Reporting Area or a Set of Core Network predefined Presence Reporting Areas. When present, it shall be encoded as a string representing an integer in the following ranges: 0 to 8 388 607 for UE-dedicated PRA 8 388 608 to 16 777 215 for Core Network predefined PRA Examples: PRA ID 123 is encoded as \\\"123\\\" PRA ID 11 238 660 is encoded as \\\"11238660\\\"")
    additionalPraId: Optional[str] = Field(None, description="This IE may be present if the praId IE is present and if it contains a PRA identifier referring to a set of Core Network predefined Presence Reporting Areas. When present, this IE shall contain a PRA Identifier of an individual PRA within the Set of Core Network predefined Presence Reporting Areas indicated by the praId IE.")
    presenceState: Optional[PresenceState] = Field(None)
    trackingAreaList: Optional[List[Tai]] = Field(None, description="Represents the list of tracking areas that constitutes the area. This IE shall be present if the subscription or  the event report is for tracking UE presence in the tracking areas. For non 3GPP access the TAI shall be the N3GPP TAI.")
    ecgiList: Optional[List[Ecgi]] = Field(None, description="Represents the list of EUTRAN cell Ids that constitutes the area. This IE shall be present if the Area of Interest subscribed is a list of EUTRAN cell Ids.")
    ncgiList: Optional[List[Ncgi]] = Field(None, description="Represents the list of NR cell Ids that constitutes the area. This IE shall be present if the Area of Interest subscribed is a list of NR cell Ids.")
    globalRanNodeIdList: Optional[List[GlobalRanNodeId]] = Field(None, description="Represents the list of NG RAN node identifiers that constitutes the area. This IE shall be present if the Area of Interest subscribed is a list of NG RAN node identifiers.")
    globaleNbIdList: Optional[List[GlobalRanNodeId]] = Field(None, description="Represents the list of eNodeB identifiers that constitutes the area. This IE shall be  present if the Area of Interest subscribed is a list of eNodeB identifiers.")


class PcfUeCallbackInfo(OpenAPIBaseModel):
    """Contains the PCF for the UE information necessary for the PCF for the PDU session to send  SM Policy Association Establishment and Termination events."""
    callbackUri: str = Field(..., description="String providing an URI formatted according to RFC 3986.")
    bindingInfo: Optional[str] = Field(None)


class PduSessionInfo(OpenAPIBaseModel):
    """indicates the DNN and S-NSSAI combination of a PDU session."""
    snssai: Optional[Snssai] = Field(None)
    dnn: str = Field(..., description="String representing a Data Network as defined in clause 9A of 3GPP TS 23.003;  it shall contain either a DNN Network Identifier, or a full DNN with both the Network  Identifier and Operator Identifier, as specified in 3GPP TS 23.003 clause 9.1.1 and 9.1.2. It shall be coded as string in which the labels are separated by dots  (e.g. \\\"Label1.Label2.Label3\\\").")


class PcfAmPolicyControlAsTimeDistributionParam(OpenAPIBaseModel):
    """Contains the 5G acess stratum time distribution parameters."""
    asTimeDistInd: Optional[bool] = Field(None)
    uuErrorBudget: Optional[int] = Field(None, description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible with the OpenAPI 'nullable: true' property.")


class AccessType(str, Enum):
    _3_GPP_ACCESS = "3GPP_ACCESS"
    NON_3_GPP_ACCESS = "NON_3GPP_ACCESS"


class UserLocation(OpenAPIBaseModel):
    """At least one of eutraLocation, nrLocation and n3gaLocation shall be present. Several of them may be present."""
    eutraLocation: Optional[EutraLocation] = Field(None)
    nrLocation: Optional[NrLocation] = Field(None)
    n3gaLocation: Optional[N3gaLocation] = Field(None)
    utraLocation: Optional[UtraLocation] = Field(None)
    geraLocation: Optional[GeraLocation] = Field(None)


class PlmnIdNid(OpenAPIBaseModel):
    """Contains the serving core network operator PLMN ID and, for an SNPN, the NID that together with the PLMN ID identifies the SNPN."""
    mcc: str = Field(..., description="Mobile Country Code part of the PLMN, comprising 3 digits, as defined in clause 9.3.3.5 of 3GPP TS 38.413.")
    mnc: str = Field(..., description="Mobile Network Code part of the PLMN, comprising 2 or 3 digits, as defined in clause 9.3.3.5 of 3GPP TS 38.413.")
    nid: Optional[str] = Field(None, description="This represents the Network Identifier, which together with a PLMN ID is used to identify an SNPN (see 3GPP TS 23.003 and 3GPP TS 23.501 clause 5.30.2.1).")


class RatType(str, Enum):
    NR = "NR"
    EUTRA = "EUTRA"
    WLAN = "WLAN"
    VIRTUAL = "VIRTUAL"
    NBIOT = "NBIOT"
    WIRELINE = "WIRELINE"
    WIRELINE_CABLE = "WIRELINE_CABLE"
    WIRELINE_BBF = "WIRELINE_BBF"
    LTE_M = "LTE-M"
    NR_U = "NR_U"
    EUTRA_U = "EUTRA_U"
    TRUSTED_N3_GA = "TRUSTED_N3GA"
    TRUSTED_WLAN = "TRUSTED_WLAN"
    UTRA = "UTRA"
    GERA = "GERA"
    NR_LEO = "NR_LEO"
    NR_MEO = "NR_MEO"
    NR_GEO = "NR_GEO"
    NR_OTHER_SAT = "NR_OTHER_SAT"
    NR_REDCAP = "NR_REDCAP"
    WB_E_UTRAN_LEO = "WB_E_UTRAN_LEO"
    WB_E_UTRAN_MEO = "WB_E_UTRAN_MEO"
    WB_E_UTRAN_GEO = "WB_E_UTRAN_GEO"
    WB_E_UTRAN_OTHERSAT = "WB_E_UTRAN_OTHERSAT"
    NB_IOT_LEO = "NB_IOT_LEO"
    NB_IOT_MEO = "NB_IOT_MEO"
    NB_IOT_GEO = "NB_IOT_GEO"
    NB_IOT_OTHERSAT = "NB_IOT_OTHERSAT"
    LTE_M_LEO = "LTE_M_LEO"
    LTE_M_MEO = "LTE_M_MEO"
    LTE_M_GEO = "LTE_M_GEO"
    LTE_M_OTHERSAT = "LTE_M_OTHERSAT"


class Snssai(OpenAPIBaseModel):
    """When Snssai needs to be converted to string (e.g. when used in maps as key), the string shall be composed of one to three digits \"sst\" optionally followed by \"-\" and 6 hexadecimal digits \"sd\"."""
    sst: int = Field(..., description="Unsigned integer, within the range 0 to 255, representing the Slice/Service Type.  It indicates the expected Network Slice behaviour in terms of features and services. Values 0 to 127 correspond to the standardized SST range. Values 128 to 255 correspond  to the Operator-specific range. See clause 28.4.2 of 3GPP TS 23.003. Standardized values are defined in clause 5.15.2.2 of 3GPP TS 23.501.")
    sd: Optional[str] = Field(None, description="3-octet string, representing the Slice Differentiator, in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits of the SD shall appear first in the string, and the character representing the 4 least significant bit of the SD shall appear last in the string. This is an optional parameter that complements the Slice/Service type(s) to allow to  differentiate amongst multiple Network Slices of the same Slice/Service type. This IE shall be absent if no SD value is associated with the SST.")


class MappingOfSnssai(OpenAPIBaseModel):
    """Contains the mapping of S-NSSAI in the serving network and the value of the home network"""
    servingSnssai: Optional[Snssai] = Field(None)
    homeSnssai: Optional[Snssai] = Field(None)


class Guami(OpenAPIBaseModel):
    """Globally Unique AMF Identifier constructed out of PLMN, Network and AMF identity."""
    plmnId: Optional[PlmnIdNid] = Field(None)
    amfId: str = Field(..., description="String identifying the AMF ID composed of AMF Region ID (8 bits), AMF Set ID (10 bits) and AMF  Pointer (6 bits) as specified in clause 2.10.1 of 3GPP TS 23.003. It is encoded as a string of  6 hexadecimal characters (i.e., 24 bits).")


class ServiceName(str, Enum):
    NNRF_NFM = "nnrf-nfm"
    NNRF_DISC = "nnrf-disc"
    NNRF_OAUTH2 = "nnrf-oauth2"
    NNRF_OAM = "nnrf-oam"
    NNRF_CMI = "nnrf-cmi"
    NUDM_SDM = "nudm-sdm"
    NUDM_UECM = "nudm-uecm"
    NUDM_UEAU = "nudm-ueau"
    NUDM_EE = "nudm-ee"
    NUDM_PP = "nudm-pp"
    NUDM_NIDDAU = "nudm-niddau"
    NUDM_MT = "nudm-mt"
    NUDM_SSAU = "nudm-ssau"
    NUDM_RSDS = "nudm-rsds"
    NUDM_UEID = "nudm-ueid"
    NUDM_OAM = "nudm-oam"
    NUDM_CMI = "nudm-cmi"
    NAMF_COMM = "namf-comm"
    NAMF_EVTS = "namf-evts"
    NAMF_MT = "namf-mt"
    NAMF_LOC = "namf-loc"
    NAMF_MBS_COMM = "namf-mbs-comm"
    NAMF_MBS_BC = "namf-mbs-bc"
    NAMF_OAM = "namf-oam"
    NAMF_CMI = "namf-cmi"
    NSMF_PDUSESSION = "nsmf-pdusession"
    NSMF_EVENT_EXPOSURE = "nsmf-event-exposure"
    NSMF_NIDD = "nsmf-nidd"
    NSMF_OAM = "nsmf-oam"
    NSMF_CMI = "nsmf-cmi"
    NAUSF_AUTH = "nausf-auth"
    NAUSF_SORPROTECTION = "nausf-sorprotection"
    NAUSF_UPUPROTECTION = "nausf-upuprotection"
    NAUSF_OAM = "nausf-oam"
    NAUSF_CMI = "nausf-cmi"
    NNEF_PFDMANAGEMENT = "nnef-pfdmanagement"
    NNEF_SMCONTEXT = "nnef-smcontext"
    NNEF_EVENTEXPOSURE = "nnef-eventexposure"
    NNEF_EAS_DEPLOYMENT_INFO = "nnef-eas-deployment-info"
    NNEF_OAM = "nnef-oam"
    NNEF_CMI = "nnef-cmi"
    field_3GPP_CP_PARAMETER_PROVISIONING = "3gpp-cp-parameter-provisioning"
    field_3GPP_DEVICE_TRIGGERING = "3gpp-device-triggering"
    field_3GPP_BDT = "3gpp-bdt"
    field_3GPP_TRAFFIC_INFLUENCE = "3gpp-traffic-influence"
    field_3GPP_CHARGEABLE_PARTY = "3gpp-chargeable-party"
    field_3GPP_AS_SESSION_WITH_QOS = "3gpp-as-session-with-qos"
    field_3GPP_PFD_MANAGEMENT = "3gpp-pfd-management"
    field_3GPP_MSISDN_LESS_MO_SMS = "3gpp-msisdn-less-mo-sms"
    field_3GPP_SERVICE_PARAMETER = "3gpp-service-parameter"
    field_3GPP_MONITORING_EVENT = "3gpp-monitoring-event"
    field_3GPP_NIDD_CONFIGURATION_TRIGGER = "3gpp-nidd-configuration-trigger"
    field_3GPP_NIDD = "3gpp-nidd"
    field_3GPP_ANALYTICSEXPOSURE = "3gpp-analyticsexposure"
    field_3GPP_RACS_PARAMETER_PROVISIONING = "3gpp-racs-parameter-provisioning"
    field_3GPP_ECR_CONTROL = "3gpp-ecr-control"
    field_3GPP_APPLYING_BDT_POLICY = "3gpp-applying-bdt-policy"
    field_3GPP_MO_LCS_NOTIFY = "3gpp-mo-lcs-notify"
    field_3GPP_TIME_SYNC = "3gpp-time-sync"
    field_3GPP_AM_INFLUENCE = "3gpp-am-influence"
    field_3GPP_AM_POLICYAUTHORIZATION = "3gpp-am-policyauthorization"
    field_3GPP_AKMA = "3gpp-akma"
    field_3GPP_EAS_DEPLOYMENT = "3gpp-eas-deployment"
    field_3GPP_IPTVCONFIGURATION = "3gpp-iptvconfiguration"
    field_3GPP_MBS_TMGI = "3gpp-mbs-tmgi"
    field_3GPP_MBS_SESSION = "3gpp-mbs-session"
    field_3GPP_AUTHENTICATION = "3gpp-authentication"
    field_3GPP_ASTI = "3gpp-asti"
    NPCF_AM_POLICY_CONTROL = "npcf-am-policy-control"
    NPCF_SMPOLICYCONTROL = "npcf-smpolicycontrol"
    NPCF_POLICYAUTHORIZATION = "npcf-policyauthorization"
    NPCF_BDTPOLICYCONTROL = "npcf-bdtpolicycontrol"
    NPCF_EVENTEXPOSURE = "npcf-eventexposure"
    NPCF_UE_POLICY_CONTROL = "npcf-ue-policy-control"
    NPCF_AM_POLICYAUTHORIZATION = "npcf-am-policyauthorization"
    NPCF_MBSPOLICYCONTROL = "npcf-mbspolicycontrol"
    NPCF_MBSPOLICYAUTH = "npcf-mbspolicyauth"
    NPCF_OAM = "npcf-oam"
    NPCF_CMI = "npcf-cmi"
    NSMSF_SMS = "nsmsf-sms"
    NNSSF_NSSELECTION = "nnssf-nsselection"
    NNSSF_NSSAIAVAILABILITY = "nnssf-nssaiavailability"
    NNSSF_OAM = "nnssf-oam"
    NNSSF_CMI = "nnssf-cmi"
    NUDR_DR = "nudr-dr"
    NUDR_GROUP_ID_MAP = "nudr-group-id-map"
    NUDR_OAM = "nudr-oam"
    NUDR_CMI = "nudr-cmi"
    NLMF_LOC = "nlmf-loc"
    N5G_EIR_EIC = "n5g-eir-eic"
    NBSF_MANAGEMENT = "nbsf-management"
    NCHF_SPENDINGLIMITCONTROL = "nchf-spendinglimitcontrol"
    NCHF_CONVERGEDCHARGING = "nchf-convergedcharging"
    NCHF_OFFLINEONLYCHARGING = "nchf-offlineonlycharging"
    NNWDAF_EVENTSSUBSCRIPTION = "nnwdaf-eventssubscription"
    NNWDAF_ANALYTICSINFO = "nnwdaf-analyticsinfo"
    NNWDAF_DATAMANAGEMENT = "nnwdaf-datamanagement"
    NNWDAF_MLMODELPROVISION = "nnwdaf-mlmodelprovision"
    NGMLC_LOC = "ngmlc-loc"
    NUCMF_PROVISIONING = "nucmf-provisioning"
    NUCMF_UECAPABILITYMANAGEMENT = "nucmf-uecapabilitymanagement"
    NHSS_SDM = "nhss-sdm"
    NHSS_UECM = "nhss-uecm"
    NHSS_UEAU = "nhss-ueau"
    NHSS_EE = "nhss-ee"
    NHSS_IMS_SDM = "nhss-ims-sdm"
    NHSS_IMS_UECM = "nhss-ims-uecm"
    NHSS_IMS_UEAU = "nhss-ims-ueau"
    NHSS_GBA_SDM = "nhss-gba-sdm"
    NHSS_GBA_UEAU = "nhss-gba-ueau"
    NSEPP_TELESCOPIC = "nsepp-telescopic"
    NSORAF_SOR = "nsoraf-sor"
    NSPAF_SECURED_PACKET = "nspaf-secured-packet"
    NUDSF_DR = "nudsf-dr"
    NUDSF_TIMER = "nudsf-timer"
    NNSSAAF_NSSAA = "nnssaaf-nssaa"
    NNSSAAF_AIW = "nnssaaf-aiw"
    NAANF_AKMA = "naanf-akma"
    N5GDDNMF_DISCOVERY = "n5gddnmf-discovery"
    NMFAF_3DADM = "nmfaf-3dadm"
    NMFAF_3CADM = "nmfaf-3cadm"
    NEASDF_DNSCONTEXT = "neasdf-dnscontext"
    NEASDF_BASELINEDNSPATTERN = "neasdf-baselinednspattern"
    NDCCF_DM = "ndccf-dm"
    NDCCF_CM = "ndccf-cm"
    NNSACF_NSAC = "nnsacf-nsac"
    NNSACF_SLICE_EE = "nnsacf-slice-ee"
    NMBSMF_TMGI = "nmbsmf-tmgi"
    NMBSMF_MBSSESSION = "nmbsmf-mbssession"
    NADRF_DM = "nadrf-dm"
    NBSP_GBA = "nbsp-gba"
    NTSCTSF_TIME_SYNC = "ntsctsf-time-sync"
    NTSCTSF_QOS_TSCAI = "ntsctsf-qos-tscai"
    NTSCTSF_ASTI = "ntsctsf-asti"
    NPKMF_KEYREQ = "npkmf-keyreq"
    NPKMF_USERID = "npkmf-userid"
    NPKMF_DISCOVERY = "npkmf-discovery"
    NMNPF_NPSTATUS = "nmnpf-npstatus"
    NIWMSC_SMSERVICE = "niwmsc-smservice"
    NMBSF_MBS_US = "nmbsf-mbs-us"
    NMBSF_MBS_UD_INGEST = "nmbsf-mbs-ud-ingest"
    NMBSTF_DISTSESSION = "nmbstf-distsession"
    NPANF_PROSEKEY = "npanf-prosekey"
    NPANF_USERID = "npanf-userid"
    NUPF_OAM = "nupf-oam"
    NUPF_CMI = "nupf-cmi"


class TraceData(OpenAPIBaseModel):
    """contains Trace control and configuration parameters."""
    traceRef: str = Field(..., description="Trace Reference (see 3GPP TS 32.422).It shall be encoded as the concatenation of MCC, MNC and Trace ID as follows: 'MCC'<MNC'-'Trace ID'The Trace ID shall be encoded as a 3 octet string in hexadecimal representation. Each character in the Trace ID string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits of the Trace ID shall appear first  in the string, and the character representing the 4 least significant bit of the Trace ID shall appear last in the string.")
    traceDepth: TraceDepth = Field(...)
    neTypeList: str = Field(..., description="List of NE Types (see 3GPP TS 32.422).It shall be encoded as an octet string in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits shall appear first in the string, and the character representing the 4 least significant bit shall appear last in the string.Octets shall be coded according to 3GPP TS 32.422.")
    eventList: str = Field(..., description="Triggering events (see 3GPP TS 32.422).It shall be encoded as an octet string in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits shall appear first in the string, and the character representing the 4 least significant bit shall appear last in the string. Octets shall be coded according to 3GPP TS 32.422.")
    collectionEntityIpv4Addr: Optional[str] = Field(None, description="String identifying a IPv4 address formatted in the 'dotted decimal' notation as defined in RFC 1166.")
    collectionEntityIpv6Addr: Optional[str] = Field(None)
    interfaceList: Optional[str] = Field(None, description="List of Interfaces (see 3GPP TS 32.422).It shall be encoded as an octet string in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits shall appear first in the string, and the character representing the  4 least significant bit shall appear last in the string. Octets shall be coded according to 3GPP TS 32.422. If this attribute is not present, all the interfaces applicable to the list of NE types indicated in the neTypeList attribute should be traced.")


class NwdafData(OpenAPIBaseModel):
    """Indicates the list of Analytic ID(s) per NWDAF instance ID used for the PDU Session consumed by the SMF."""
    nwdafInstanceId: str = Field(..., description="String uniquely identifying a NF instance. The format of the NF Instance ID shall be a  Universally Unique Identifier (UUID) version 4, as described in IETF RFC 4122.")
    nwdafEvents: Optional[List[NwdafEvent]] = Field(None)


class RestrictionType(str, Enum):
    ALLOWED_AREAS = "ALLOWED_AREAS"
    NOT_ALLOWED_AREAS = "NOT_ALLOWED_AREAS"


class Area(OpenAPIBaseModel):
    """Provides area information."""
    tacs: Optional[List[str]] = Field(None)
    areaCode: Optional[str] = Field(None, description="Values are operator specific.")


class WirelineArea(OpenAPIBaseModel):
    """One and only one of the \"globLineIds\", \"hfcNIds\", \"areaCodeB\" and \"areaCodeC\" attributes shall be included in a WirelineArea data structure"""
    globalLineIds: Optional[List[str]] = Field(None)
    hfcNIds: Optional[List[str]] = Field(None)
    areaCodeB: Optional[str] = Field(None, description="Values are operator specific.")
    areaCodeC: Optional[str] = Field(None, description="Values are operator specific.")


class CandidateForReplacement(OpenAPIBaseModel):
    """Represents a list of candidate DNNs for replacement for an S-NSSAI."""
    snssai: Optional[Snssai] = Field(None)
    dnns: Optional[List[str]] = Field(None)


class SliceMbr(OpenAPIBaseModel):
    """MBR related to slice"""
    uplink: str = Field(..., description="String representing a bit rate; the prefixes follow the standard symbols from The International System of Units, and represent x1000 multipliers, with the exception that prefix \\\"K\\\" is used to represent the standard symbol \\\"k\\\".")
    downlink: str = Field(..., description="String representing a bit rate; the prefixes follow the standard symbols from The International System of Units, and represent x1000 multipliers, with the exception that prefix \\\"K\\\" is used to represent the standard symbol \\\"k\\\".")


class PresenceState(str, Enum):
    IN_AREA = "IN_AREA"
    OUT_OF_AREA = "OUT_OF_AREA"
    UNKNOWN = "UNKNOWN"
    INACTIVE = "INACTIVE"


class Tai(OpenAPIBaseModel):
    """Contains the tracking area identity as described in 3GPP 23.003"""
    plmnId: Optional[PlmnId] = Field(None)
    tac: str = Field(..., description="2 or 3-octet string identifying a tracking area code as specified in clause 9.3.3.10  of 3GPP TS 38.413, in hexadecimal representation. Each character in the string shall  take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the 4 most significant bits of the TAC shall  appear first in the string, and the character representing the 4 least significant bit  of the TAC shall appear last in the string.")
    nid: Optional[str] = Field(None, description="This represents the Network Identifier, which together with a PLMN ID is used to identify an SNPN (see 3GPP TS 23.003 and 3GPP TS 23.501 clause 5.30.2.1).")


class Ecgi(OpenAPIBaseModel):
    """Contains the ECGI (E-UTRAN Cell Global Identity), as described in 3GPP 23.003"""
    plmnId: Optional[PlmnId] = Field(None)
    eutraCellId: str = Field(..., description="28-bit string identifying an E-UTRA Cell Id as specified in clause 9.3.1.9 of  3GPP TS 38.413, in hexadecimal representation. Each character in the string shall take a  value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most  significant character representing the 4 most significant bits of the Cell Id shall appear  first in the string, and the character representing the 4 least significant bit of the  Cell Id shall appear last in the string.")
    nid: Optional[str] = Field(None, description="This represents the Network Identifier, which together with a PLMN ID is used to identify an SNPN (see 3GPP TS 23.003 and 3GPP TS 23.501 clause 5.30.2.1).")


class Ncgi(OpenAPIBaseModel):
    """Contains the NCGI (NR Cell Global Identity), as described in 3GPP 23.003"""
    plmnId: Optional[PlmnId] = Field(None)
    nrCellId: str = Field(..., description="36-bit string identifying an NR Cell Id as specified in clause 9.3.1.7 of 3GPP TS 38.413,  in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\",  \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character  representing the 4 most significant bits of the Cell Id shall appear first in the string, and  the character representing the 4 least significant bit of the Cell Id shall appear last in the  string.")
    nid: Optional[str] = Field(None, description="This represents the Network Identifier, which together with a PLMN ID is used to identify an SNPN (see 3GPP TS 23.003 and 3GPP TS 23.501 clause 5.30.2.1).")


class GlobalRanNodeId(OpenAPIBaseModel):
    """One of the six attributes n3IwfId, gNbIdm, ngeNbId, wagfId, tngfId, eNbId shall be present."""
    plmnId: Optional[PlmnId] = Field(None)
    n3IwfId: Optional[str] = Field(None, description="This represents the identifier of the N3IWF ID as specified in clause 9.3.1.57 of  3GPP TS 38.413 in hexadecimal representation. Each character in the string shall take a value  of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant  character representing the 4 most significant bits of the N3IWF ID shall appear first in the  string, and the character representing the 4 least significant bit of the N3IWF ID shall  appear last in the string.")
    gNbId: Optional[GNbId] = Field(None)
    ngeNbId: Optional[str] = Field(None, description="This represents the identifier of the ng-eNB ID as specified in clause 9.3.1.8 of  3GPP TS 38.413. The value of the ng-eNB ID shall be encoded in hexadecimal representation.  Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and  shall represent 4 bits. The padding 0 shall be added to make multiple nibbles, so the most  significant character representing the padding 0 if required together with the 4 most  significant bits of the ng-eNB ID shall appear first in the string, and the character  representing the 4 least significant bit of the ng-eNB ID (to form a nibble) shall appear last  in the string.")
    wagfId: Optional[str] = Field(None, description="This represents the identifier of the W-AGF ID as specified in clause 9.3.1.162 of  3GPP TS 38.413 in hexadecimal representation. Each character in the string shall take a value  of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant  character representing the 4 most significant bits of the W-AGF ID shall appear first in the  string, and the character representing the 4 least significant bit of the W-AGF ID shall  appear last in the string.")
    tngfId: Optional[str] = Field(None, description="This represents the identifier of the TNGF ID as specified in clause 9.3.1.161 of  3GPP TS 38.413  in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\"  to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant character representing the  4 most significant bits of the TNGF ID shall appear first in the string, and the character  representing the 4 least significant bit of the TNGF ID shall appear last in the string.")
    twifId: Optional[str] = Field(None)
    nid: Optional[str] = Field(None, description="This represents the Network Identifier, which together with a PLMN ID is used to identify an SNPN (see 3GPP TS 23.003 and 3GPP TS 23.501 clause 5.30.2.1).")
    eNbId: Optional[str] = Field(None, description="This represents the identifier of the eNB ID as specified in clause 9.2.1.37 of  3GPP TS 36.413. The string shall be formatted with the following pattern  '^('MacroeNB-[A-Fa-f0-9]{5}|LMacroeNB-[A-Fa-f0-9]{6}|SMacroeNB-[A-Fa-f0-9]{5} |HomeeNB-[A-Fa-f0-9]{7})$'. The value of the eNB ID shall be encoded in hexadecimal representation. Each character in the  string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits.  The padding 0 shall be added to make multiple nibbles, so the most significant character  representing the padding 0 if required together with the 4 most significant bits of the eNB ID  shall appear first in the string, and the character representing the 4 least significant bit  of the eNB ID (to form a nibble) shall appear last in the string.")


class EutraLocation(OpenAPIBaseModel):
    """Contains the E-UTRA user location."""
    tai: Optional[Tai] = Field(None)
    ignoreTai: Optional[bool] = Field(None)
    ecgi: Optional[Ecgi] = Field(None)
    ignoreEcgi: Optional[bool] = Field(None, description="This flag when present shall indicate that the Ecgi shall be ignored When present, it shall be set as follows: - true: ecgi shall be ignored. - false (default): ecgi shall not be ignored.")
    ageOfLocationInformation: Optional[int] = Field(None, description="The value represents the elapsed time in minutes since the last network contact of the mobile station.  Value \\\"0\\\" indicates that the location information was obtained after a successful paging procedure for Active Location Retrieval when the UE is in idle mode or after a successful NG-RAN location reporting procedure with the eNB when the UE is in connected mode.  Any other value than \\\"0\\\" indicates that the location information is the last known one.  See 3GPP TS 29.002 clause 17.7.8.")
    ueLocationTimestamp: Optional[datetime] = Field(None, description="string with format 'date-time' as defined in OpenAPI.")
    geographicalInformation: Optional[str] = Field(None, description="Refer to geographical Information. See 3GPP TS 23.032 clause 7.3.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    geodeticInformation: Optional[str] = Field(None, description="Refers to Calling Geodetic Location. See ITU-T Recommendation Q.763 (1999) [24] clause 3.88.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    globalNgenbId: Optional[GlobalRanNodeId] = Field(None)
    globalENbId: Optional[GlobalRanNodeId] = Field(None)


class NrLocation(OpenAPIBaseModel):
    """Contains the NR user location."""
    tai: Optional[Tai] = Field(None)
    ncgi: Optional[Ncgi] = Field(None)
    ignoreNcgi: Optional[bool] = Field(None)
    ageOfLocationInformation: Optional[int] = Field(None, description="The value represents the elapsed time in minutes since the last network contact of the mobile station. Value \\\"0\\\" indicates that the location information was obtained after a successful paging procedure for Active Location Retrieval when the UE is in idle mode or after a successful  NG-RAN location reporting procedure with the eNB when the UE is in connected mode. Any other value than \\\"0\\\" indicates that the location information is the last known one. See 3GPP TS 29.002 clause 17.7.8.")
    ueLocationTimestamp: Optional[datetime] = Field(None, description="string with format 'date-time' as defined in OpenAPI.")
    geographicalInformation: Optional[str] = Field(None, description="Refer to geographical Information. See 3GPP TS 23.032 clause 7.3.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    geodeticInformation: Optional[str] = Field(None, description="Refers to Calling Geodetic Location. See ITU-T Recommendation Q.763 (1999) [24] clause 3.88.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    globalGnbId: Optional[GlobalRanNodeId] = Field(None)


class N3gaLocation(OpenAPIBaseModel):
    """Contains the Non-3GPP access user location."""
    n3gppTai: Optional[Tai] = Field(None)
    n3IwfId: Optional[str] = Field(None, description="This IE shall contain the N3IWF identifier received over NGAP and shall be encoded as a  string of hexadecimal characters. Each character in the string shall take a value of \\\"0\\\"  to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The most significant  character representing the 4 most significant bits of the N3IWF ID shall appear first in  the string, and the character representing the 4 least significant bit of the N3IWF ID  shall appear last in the string.")
    ueIpv4Addr: Optional[str] = Field(None, description="String identifying a IPv4 address formatted in the 'dotted decimal' notation as defined in RFC 1166.")
    ueIpv6Addr: Optional[str] = Field(None)
    portNumber: Optional[int] = Field(None, description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible.")
    protocol: Optional[TransportProtocol] = Field(None)
    tnapId: Optional[TnapId] = Field(None)
    twapId: Optional[TwapId] = Field(None)
    hfcNodeId: Optional[HfcNodeId] = Field(None)
    gli: Optional[str] = Field(None, description="string with format 'bytes' as defined in OpenAPI")
    w5gbanLineType: Optional[LineType] = Field(None)
    gci: Optional[str] = Field(None, description="Global Cable Identifier uniquely identifying the connection between the 5G-CRG or FN-CRG to the 5GS. See clause 28.15.4 of 3GPP TS 23.003. This shall be encoded as a string per clause 28.15.4 of 3GPP TS 23.003, and compliant with the syntax specified  in clause 2.2  of IETF RFC 7542 for the username part of a NAI. The GCI value is specified in CableLabs WR-TR-5WWC-ARCH.")


class UtraLocation(OpenAPIBaseModel):
    """Exactly one of cgi, sai or lai shall be present."""
    cgi: Optional[CellGlobalId] = Field(None)
    sai: Optional[ServiceAreaId] = Field(None)
    lai: Optional[LocationAreaId] = Field(None)
    rai: Optional[RoutingAreaId] = Field(None)
    ageOfLocationInformation: Optional[int] = Field(None, description="The value represents the elapsed time in minutes since the last network contact of the mobile station.  Value \\\"0\\\" indicates that the location information was obtained after a successful paging procedure for  Active Location Retrieval when the UE is in idle mode  or after a successful location reporting procedure  the UE is in connected mode. Any other value than \\\"0\\\" indicates that the location information is the last known one.  See 3GPP TS 29.002 clause 17.7.8.")
    ueLocationTimestamp: Optional[datetime] = Field(None, description="string with format 'date-time' as defined in OpenAPI.")
    geographicalInformation: Optional[str] = Field(None, description="Refer to geographical Information.See 3GPP TS 23.032 clause 7.3.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    geodeticInformation: Optional[str] = Field(None, description="Refers to Calling Geodetic Location. See ITU-T Recommendation Q.763 (1999) clause 3.88.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")


class GeraLocation(OpenAPIBaseModel):
    """Exactly one of cgi, sai or lai shall be present."""
    locationNumber: Optional[str] = Field(None, description="Location number within the PLMN. See 3GPP TS 23.003, clause 4.5.")
    cgi: Optional[CellGlobalId] = Field(None)
    rai: Optional[RoutingAreaId] = Field(None)
    sai: Optional[ServiceAreaId] = Field(None)
    lai: Optional[LocationAreaId] = Field(None)
    vlrNumber: Optional[str] = Field(None, description="VLR number. See 3GPP TS 23.003 clause 5.1.")
    mscNumber: Optional[str] = Field(None, description="MSC number. See 3GPP TS 23.003 clause 5.1.")
    ageOfLocationInformation: Optional[int] = Field(None, description="The value represents the elapsed time in minutes since the last network contact of the mobile station. Value \\\"0\\\" indicates that the location information was obtained after a successful paging procedure for  Active Location Retrieval when the UE is in idle mode or after a successful location reporting procedure the UE is in connected mode. Any other value than \\\"0\\\" indicates that the location information is the last known one. See 3GPP TS 29.002 clause 17.7.8.")
    ueLocationTimestamp: Optional[datetime] = Field(None, description="string with format 'date-time' as defined in OpenAPI.")
    geographicalInformation: Optional[str] = Field(None, description="Refer to geographical Information.See 3GPP TS 23.032 clause 7.3.2. Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")
    geodeticInformation: Optional[str] = Field(None, description="Refers to Calling Geodetic Location.See ITU-T Recommendation Q.763 (1999) clause 3.88.2.  Only the description of an ellipsoid point with uncertainty circle is allowed to be used.")


class TraceDepth(str, Enum):
    MINIMUM = "MINIMUM"
    MEDIUM = "MEDIUM"
    MAXIMUM = "MAXIMUM"
    MINIMUM_WO_VENDOR_EXTENSION = "MINIMUM_WO_VENDOR_EXTENSION"
    MEDIUM_WO_VENDOR_EXTENSION = "MEDIUM_WO_VENDOR_EXTENSION"
    MAXIMUM_WO_VENDOR_EXTENSION = "MAXIMUM_WO_VENDOR_EXTENSION"


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


class PlmnId(OpenAPIBaseModel):
    """When PlmnId needs to be converted to string (e.g. when used in maps as key), the string  shall be composed of three digits \"mcc\" followed by \"-\" and two or three digits \"mnc\"."""
    mcc: str = Field(..., description="Mobile Country Code part of the PLMN, comprising 3 digits, as defined in clause 9.3.3.5 of 3GPP TS 38.413.")
    mnc: str = Field(..., description="Mobile Network Code part of the PLMN, comprising 2 or 3 digits, as defined in clause 9.3.3.5 of 3GPP TS 38.413.")


class GNbId(OpenAPIBaseModel):
    """Provides the G-NB identifier."""
    bitLength: int = Field(..., description="Unsigned integer representing the bit length of the gNB ID as defined in clause 9.3.1.6 of 3GPP TS 38.413 [11], within the range 22 to 32.")
    gNBValue: str = Field(..., description="This represents the identifier of the gNB. The value of the gNB ID shall be encoded in hexadecimal representation. Each character in the string shall take a value of \\\"0\\\" to \\\"9\\\", \\\"a\\\" to \\\"f\\\" or \\\"A\\\" to \\\"F\\\" and shall represent 4 bits. The padding 0 shall be added to make multiple nibbles,  the most significant character representing the padding 0 if required together with the 4 most significant bits of the gNB ID shall appear first in the string, and the character representing the 4 least significant bit of the gNB ID shall appear last in the string.")


class TransportProtocol(str, Enum):
    UDP = "UDP"
    TCP = "TCP"


class TnapId(OpenAPIBaseModel):
    """Contain the TNAP Identifier see clause5.6.2 of 3GPP TS 23.501."""
    ssId: Optional[str] = Field(None, description="This IE shall be present if the UE is accessing the 5GC via a trusted WLAN access network.When present, it shall contain the SSID of the access point to which the UE is attached, that is received over NGAP,  see IEEE Std 802.11-2012.")
    bssId: Optional[str] = Field(None, description="When present, it shall contain the BSSID of the access point to which the UE is attached, that is received over NGAP, see IEEE Std 802.11-2012.")
    civicAddress: Optional[str] = Field(None, description="string with format 'bytes' as defined in OpenAPI")


class TwapId(OpenAPIBaseModel):
    """Contain the TWAP Identifier as defined in clause 4.2.8.5.3 of 3GPP TS 23.501 or the WLAN location information as defined in clause 4.5.7.2.8 of 3GPP TS 23.402."""
    ssId: str = Field(..., description="This IE shall contain the SSID of the access point to which the UE is attached, that is received over NGAP, see IEEE Std 802.11-2012.")
    bssId: Optional[str] = Field(None, description="When present, it shall contain the BSSID of the access point to which the UE is attached, for trusted WLAN access, see IEEE Std 802.11-2012.")
    civicAddress: Optional[str] = Field(None, description="string with format 'bytes' as defined in OpenAPI")


class HfcNodeId(OpenAPIBaseModel):
    """REpresents the HFC Node Identifer received over NGAP."""
    hfcNId: str = Field(..., description="This IE represents the identifier of the HFC node Id as specified in CableLabs WR-TR-5WWC-ARCH. It is provisioned by the wireline operator as part of wireline operations and may contain up to six characters.")


class LineType(str, Enum):
    DSL = "DSL"
    PON = "PON"


class CellGlobalId(OpenAPIBaseModel):
    """Contains a Cell Global Identification as defined in 3GPP TS 23.003, clause 4.3.1."""
    plmnId: Optional[PlmnId] = Field(None)
    lac: str = Field(...)
    cellId: str = Field(...)


class ServiceAreaId(OpenAPIBaseModel):
    """Contains a Service Area Identifier as defined in 3GPP TS 23.003, clause 12.5."""
    plmnId: Optional[PlmnId] = Field(None)
    lac: str = Field(..., description="Location Area Code.")
    sac: str = Field(..., description="Service Area Code.")


class LocationAreaId(OpenAPIBaseModel):
    """Contains a Location area identification as defined in 3GPP TS 23.003, clause 4.1."""
    plmnId: Optional[PlmnId] = Field(None)
    lac: str = Field(..., description="Location Area Code.")


class RoutingAreaId(OpenAPIBaseModel):
    """Contains a Routing Area Identification as defined in 3GPP TS 23.003, clause 4.2."""
    plmnId: Optional[PlmnId] = Field(None)
    lac: str = Field(..., description="Location Area Code")
    rac: str = Field(..., description="Routing Area Code")


# Resolve forward references across generated models.
PcfAmPolicyControlPolicyAssociation.model_rebuild()
PcfAmPolicyControlPolicyAssociationRequest.model_rebuild()
ServiceAreaRestriction.model_rebuild()
WirelineServiceAreaRestriction.model_rebuild()
SmfSelectionData.model_rebuild()
Ambr.model_rebuild()
UeSliceMbr.model_rebuild()
PresenceInfo.model_rebuild()
PcfUeCallbackInfo.model_rebuild()
PduSessionInfo.model_rebuild()
PcfAmPolicyControlAsTimeDistributionParam.model_rebuild()
UserLocation.model_rebuild()
PlmnIdNid.model_rebuild()
Snssai.model_rebuild()
MappingOfSnssai.model_rebuild()
Guami.model_rebuild()
TraceData.model_rebuild()
NwdafData.model_rebuild()
Area.model_rebuild()
WirelineArea.model_rebuild()
CandidateForReplacement.model_rebuild()
SliceMbr.model_rebuild()
Tai.model_rebuild()
Ecgi.model_rebuild()
Ncgi.model_rebuild()
GlobalRanNodeId.model_rebuild()
EutraLocation.model_rebuild()
NrLocation.model_rebuild()
N3gaLocation.model_rebuild()
UtraLocation.model_rebuild()
GeraLocation.model_rebuild()
PlmnId.model_rebuild()
GNbId.model_rebuild()
TnapId.model_rebuild()
TwapId.model_rebuild()
HfcNodeId.model_rebuild()
CellGlobalId.model_rebuild()
ServiceAreaId.model_rebuild()
LocationAreaId.model_rebuild()
RoutingAreaId.model_rebuild()


__all__ = ["PcfAmPolicyControlPolicyAssociation", "PcfAmPolicyControlPolicyAssociationRequest", "PcfAmPolicyControlRequestTrigger", "ServiceAreaRestriction", "WirelineServiceAreaRestriction", "SmfSelectionData", "Ambr", "UeSliceMbr", "PresenceInfo", "PcfUeCallbackInfo", "PduSessionInfo", "PcfAmPolicyControlAsTimeDistributionParam", "AccessType", "UserLocation", "PlmnIdNid", "RatType", "Snssai", "MappingOfSnssai", "Guami", "ServiceName", "TraceData", "NwdafData", "RestrictionType", "Area", "WirelineArea", "CandidateForReplacement", "SliceMbr", "PresenceState", "Tai", "Ecgi", "Ncgi", "GlobalRanNodeId", "EutraLocation", "NrLocation", "N3gaLocation", "UtraLocation", "GeraLocation", "TraceDepth", "NwdafEvent", "PlmnId", "GNbId", "TransportProtocol", "TnapId", "TwapId", "HfcNodeId", "LineType", "CellGlobalId", "ServiceAreaId", "LocationAreaId", "RoutingAreaId"]
