from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .PcfAmPolicyControl import (
    AccessType,
    Ambr,
    Guami,
    MappingOfSnssai,
    PcfAmPolicyControlPolicyAssociation,
    PlmnIdNid,
    PresenceInfo,
    RatType,
    ServiceAreaRestriction,
    SmfSelectionData,
    Snssai,
    UserLocation,
    WirelineServiceAreaRestriction,
)
from .SmPolicyContextData import SmPolicyContextData
from .SmPolicyDecision import SmPolicyDecision


class IdentityContext(BaseModel):
    supi: str
    gpsi: Optional[str] = None
    pei: Optional[str] = None
    groupIds: List[str] = Field(default_factory=list)


class AccessMobilityContext(BaseModel):
    accessType: Optional[AccessType] = None
    accessTypes: List[AccessType] = Field(default_factory=list)
    ratType: Optional[RatType] = None
    ratTypes: List[RatType] = Field(default_factory=list)
    userLoc: Optional[UserLocation] = None
    guami: Optional[Guami] = None
    servingPlmn: Optional[PlmnIdNid] = None
    timeZone: Optional[str] = None
    presenceAreas: Dict[str, PresenceInfo] = Field(default_factory=dict)
    mobilityEventType: Optional[str] = None


class ServingNfContext(BaseModel):
    pcf_id: Optional[str] = None
    pcf_uri: Optional[str] = None
    amf_id: Optional[str] = None
    amf_uri: Optional[str] = None
    smf_id: Optional[str] = None
    smf_uri: Optional[str] = None
    binding_info: Dict[str, Any] = Field(default_factory=dict)


class MobilitySummary(BaseModel):
    currentAssociationId: Optional[str] = None
    currentTriggers: List[str] = Field(default_factory=list)
    lastMobilityEventType: Optional[str] = None
    lastServedTai: Optional[str] = None
    currentRfsp: Optional[int] = None
    mobilityRiskScore: Optional[float] = None
    lastUpdatedReason: Optional[str] = None


class SessionPolicyRuntimeState(BaseModel):
    packFiltIdGenerator: Optional[int] = None
    pccRuleIdGenerator: Optional[int] = None
    chargingIdGenerator: Optional[int] = None
    packFiltMapToPccRuleId: Dict[str, str] = Field(default_factory=dict)
    remainGbrUL: Optional[float] = None
    remainGbrDL: Optional[float] = None
    appSessions: Dict[str, bool] = Field(default_factory=dict)
    influenceDataToPccRule: Dict[str, str] = Field(default_factory=dict)
    subscriptionID: Optional[str] = None
    bsfBindingId: Optional[str] = None


class UeSmPolicyData(BaseModel):
    runtimeState: SessionPolicyRuntimeState = Field(default_factory=SessionPolicyRuntimeState)
    policyContext: Optional[SmPolicyContextData] = None
    policyDecision: Optional[SmPolicyDecision] = None


class SessionPolicyContext(BaseModel):
    smPolicyData: Dict[str, UeSmPolicyData] = Field(default_factory=dict)
    pccRules: Dict[str, Any] = Field(default_factory=dict)
    qosDecs: Dict[str, Any] = Field(default_factory=dict)
    sessRules: Dict[str, Any] = Field(default_factory=dict)
    traffContDecs: Dict[str, Any] = Field(default_factory=dict)
    chgDecs: Dict[str, Any] = Field(default_factory=dict)
    urspRules: Dict[str, Any] = Field(default_factory=dict)
    appCatalog: List[Dict[str, Any]] = Field(default_factory=list)
    flowCatalog: List[Dict[str, Any]] = Field(default_factory=list)


class AmPolicyContext(BaseModel):
    polAssociationIDGenerator: Optional[int] = None
    associations: Dict[str, PcfAmPolicyControlPolicyAssociation] = Field(default_factory=dict)
    allowedSnssais: List[Snssai] = Field(default_factory=list)
    targetSnssais: List[Snssai] = Field(default_factory=list)
    mappingSnssais: List[MappingOfSnssai] = Field(default_factory=list)
    servAreaRes: Optional[ServiceAreaRestriction] = None
    wlServAreaRes: Optional[WirelineServiceAreaRestriction] = None
    rfsp: Optional[int] = None
    smfSelInfo: Optional[SmfSelectionData] = None
    ueAmbr: Optional[Ambr] = None
    ueSliceMbrs: List[Dict[str, Any]] = Field(default_factory=list)
    pras: Dict[str, PresenceInfo] = Field(default_factory=dict)


class UeContext(BaseModel):
    identity_context: IdentityContext
    access_mobility_context: AccessMobilityContext = Field(default_factory=AccessMobilityContext)
    am_policy_context: AmPolicyContext = Field(default_factory=AmPolicyContext)
    session_policy_context: SessionPolicyContext = Field(default_factory=SessionPolicyContext)
    serving_nf_context: ServingNfContext = Field(default_factory=ServingNfContext)
    mobility_summary: MobilitySummary = Field(default_factory=MobilitySummary)
    udrUri: Optional[str] = None
    afRoutReq: Optional[Dict[str, Any]] = None
    aspId: Optional[str] = None
    ratingGroupData: Dict[str, List[int]] = Field(default_factory=dict)

    @property
    def supi(self) -> str:
        return self.identity_context.supi

