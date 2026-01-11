from typing import List, Optional, Dict, Any, ForwardRef
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from .SmPolicyContextData import (
    SmPolicyContextData, 
    UserLocation, 
    PresenceInfo, 
    AccessType, 
    PlmnIdNid, 
    Guami,
    Snssai
)
from .SmPolicyDecision import SmPolicyDecision


# --- Placeholder Models for missing dependencies ---
# These models appeared in the Go code but definitions were not provided in recent conversions.
# We define them as basic Pydantic models to allow the code to be valid.

class ServiceName(str, Enum):
    # Add values as needed
    NNRF_DISC = "nnrf-disc"
    NNRF_NFM = "nnrf-nfm"
    NPARAM_HPARAM_PROVISIONING = "nparam-hparam-provisioning"
    # ... placeholders
    pass

class PcfAmPolicyControlRequestTrigger(str, Enum):
    # Add values as needed
    PLMN_CH = "PLMN_CH"
    RES_MO_RE = "RES_MO_RE"
    AC_TY_CH = "AC_TY_CH"
    UE_IP_CH = "UE_IP_CH"
    UE_MAC_CH = "UE_MAC_CH"
    AN_CH_COR = "AN_CH_COR"
    # ... placeholders
    pass

class ServiceAreaRestriction(BaseModel):
    restrictionType: Optional[str] = None
    areas: Optional[List[Dict[str, Any]]] = None
    maxNumOfTAs: Optional[int] = None
    maxNumOfTAsForNotAllowedAreas: Optional[int] = None
    pass

class AmPolicyData(BaseModel):
    # Placeholder for AmPolicyData
    # Contains Subscription Data
    hueSupi: Optional[str] = None
    subscCats: Optional[List[str]] = None
    pass

class PolicyDataSubscription(BaseModel):
    # Placeholder
    notificationUri: str
    monitoredResourceUris: List[str]
    pass

class PolicyDataChangeNotification(BaseModel):
    # Placeholder
    amPolicyData: Optional[AmPolicyData] = None
    uePolicySet: Optional[Any] = None
    plmnId: Optional[PlmnIdNid] = None
    usageMonData: Optional[Any] = None
    pass

class AfRoutingRequirement(BaseModel):
    # Based on Af_routing_requirement.py (which was JSON templates)
    # We define a loose structure here or strict if we parsed the JSON template.
    # Given the previous file was just templates, we treat this as a Dict or generic model.
    appReloc: Optional[bool] = None
    routeToLocs: Optional[List[Dict[str, Any]]] = None
    spVal: Optional[Dict[str, Any]] = None
    tempVals: Optional[List[Dict[str, Any]]] = None
    upPathChgSub: Optional[Dict[str, Any]] = None
    addrPreserInd: Optional[bool] = None
    pass

class AppSessionIdStore(BaseModel):
    # Placeholder for the internal store
    # Likely a wrapper around a map or list
    pass

# --- SmPolicyData Conversion (from what was seen in Sm_policy_data.py) ---

class UsageMonDataLimit(BaseModel):
    limitId: str
    scopes: Optional[Dict[str, Any]] = None
    umLevel: Optional[str] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    usageLimit: Optional[int] = None
    resetPeriod: Optional[str] = None

class UsageMonData(BaseModel):
    limitId: str
    scopes: Optional[Dict[str, Any]] = None
    umLevel: Optional[str] = None
    totVol: Optional[int] = None
    totVolUl: Optional[int] = None
    totVolDl: Optional[int] = None
    totTime: Optional[int] = None

class SmPolicySnssaiData(BaseModel):
    snssai: Optional[Snssai] = None
    smPolicyDnnData: Optional[Dict[str, Any]] = None

class SmPolicyData(BaseModel):
    # Contains Session Management Policy data per S-NSSAI
    smPolicySnssaiData: Optional[Dict[str, SmPolicySnssaiData]] = None
    # Contains a list of usage monitoring profiles
    umDataLimits: Optional[Dict[str, UsageMonDataLimit]] = None
    # Contains the remaining allowed usage data
    umData: Optional[Dict[str, UsageMonData]] = None
    suppFeat: Optional[str] = None
    resetIds: Optional[List[str]] = None

# --- Main Ue_context_pcf Models ---

class UeAMPolicyData(BaseModel):
    polAssoId: str
    accessType: Optional[AccessType] = None
    notificationUri: Optional[str] = None
    servingPlmn: Optional[PlmnIdNid] = None
    altNotifIpv4Addrs: Optional[List[str]] = None
    altNotifIpv6Addrs: Optional[List[str]] = None
    # TODO: AMF Status Change
    amfStatusUri: Optional[str] = None
    guami: Optional[Guami] = None
    serviceName: Optional[ServiceName] = None
    # TraceReq *TraceData
    # Policy Association
    triggers: Optional[List[PcfAmPolicyControlRequestTrigger]] = None
    servAreaRes: Optional[ServiceAreaRestriction] = None
    rfsp: Optional[int] = None
    userLoc: Optional[UserLocation] = None
    timeZone: Optional[str] = None
    suppFeat: Optional[str] = None
    # about AF request
    pras: Optional[Dict[str, PresenceInfo]] = None
    # related to UDR Subscription Data
    amPolicyData: Optional[AmPolicyData] = None
    # Corresponding UE
    pcfUe: Optional['UeContext'] = None # Reference to parent

class UeSmPolicyData(BaseModel):
    # PackFiltIdGenerator int32
    packFiltIdGenerator: Optional[int] = None
    pccRuleIdGenerator: Optional[int] = None
    chargingIdGenerator: Optional[int] = None

    # FlowMapsToPackFiltIds  map[string][]string
    # packFiltMapToPccRuleId map[string]string
    packFiltMapToPccRuleId: Optional[Dict[str, str]] = None
    
    # Related to GBR
    remainGbrUL: Optional[float] = None
    remainGbrDL: Optional[float] = None
    
    # related to UDR Subscription Data
    smPolicyData: Optional[SmPolicyData] = None
    
    # related to Policy
    policyContext: Optional[SmPolicyContextData] = None
    policyDecision: Optional[SmPolicyDecision] = None
    
    # related to AppSession
    appSessions: Optional[Dict[str, bool]] = None
    
    # Corresponding UE
    pcfUe: Optional['UeContext'] = None
    
    influenceDataToPccRule: Optional[Dict[str, str]] = None
    subscriptionID: Optional[str] = None
    
    # BSF Integration
    bsfBindingId: Optional[str] = None

class UeContext(BaseModel):
    # Ue Context
    supi: str
    gpsi: Optional[str] = None
    pei: Optional[str] = None
    groupIds: Optional[List[str]] = None
    polAssociationIDGenerator: Optional[int] = None
    
    # use PolAssoId(ue.Supi-numPolId) as key
    amPolicyData: Optional[Dict[str, UeAMPolicyData]] = None

    # Udr Ref
    udrUri: Optional[str] = None
    
    # SMPolicy
    # use smPolicyId(ue.Supi-pduSessionId) as key
    smPolicyData: Optional[Dict[str, UeSmPolicyData]] = None
    
    # App Session Related
    # AppSessionIDGenerator is a struct pointer in Go, tricky to map directly without logic.
    # We map it to an int or a complex object if needed.
    appSessionIDGenerator: Any = None 
    
    # PolicyAuth
    afRoutReq: Optional[AfRoutingRequirement] = None
    aspId: Optional[str] = None
    
    # Policy Decision
    appSessionIdStore: Optional[AppSessionIdStore] = None
    policyDataSubscriptionStore: Optional[PolicyDataSubscription] = None
    policyDataChangeStore: Optional[PolicyDataChangeNotification] = None

    # ChargingRatingGroup
    # use smPolicyId(ue.Supi-pduSessionId) as key
    ratingGroupData: Optional[Dict[str, List[int]]] = None