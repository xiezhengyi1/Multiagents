from __future__ import annotations

from typing import Any

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
    gpsi: str | None = None
    pei: str | None = None
    groupIds: list[str] = Field(default_factory=list)


class AccessMobilityContext(BaseModel):
    accessType: AccessType | None = None
    accessTypes: list[AccessType] = Field(default_factory=list)
    ratType: RatType | None = None
    ratTypes: list[RatType] = Field(default_factory=list)
    userLoc: UserLocation | None = None
    guami: Guami | None = None
    servingPlmn: PlmnIdNid | None = None
    timeZone: str | None = None
    presenceAreas: dict[str, PresenceInfo] = Field(default_factory=dict)
    mobilityEventType: str | None = None


class ServingNfContext(BaseModel):
    pcf_id: str | None = None
    pcf_uri: str | None = None
    amf_id: str | None = None
    amf_uri: str | None = None
    smf_id: str | None = None
    smf_uri: str | None = None
    binding_info: dict[str, Any] = Field(default_factory=dict)


class MobilitySummary(BaseModel):
    currentAssociationId: str | None = None
    currentTriggers: list[str] = Field(default_factory=list)
    lastMobilityEventType: str | None = None
    lastServedTai: str | None = None
    currentRfsp: int | None = None
    mobilityRiskScore: float | None = None
    lastUpdatedReason: str | None = None


class SessionPolicyRuntimeState(BaseModel):
    packFiltIdGenerator: int | None = None
    pccRuleIdGenerator: int | None = None
    chargingIdGenerator: int | None = None
    packFiltMapToPccRuleId: dict[str, str] = Field(default_factory=dict)
    remainGbrUL: float | None = None
    remainGbrDL: float | None = None
    appSessions: dict[str, bool] = Field(default_factory=dict)
    influenceDataToPccRule: dict[str, str] = Field(default_factory=dict)
    subscriptionID: str | None = None
    bsfBindingId: str | None = None


class UeSmPolicyData(BaseModel):
    runtimeState: SessionPolicyRuntimeState = Field(default_factory=SessionPolicyRuntimeState)
    policyContext: SmPolicyContextData | None = None
    policyDecision: SmPolicyDecision | None = None


class SessionPolicyContext(BaseModel):
    smPolicyData: dict[str, UeSmPolicyData] = Field(default_factory=dict)
    pccRules: dict[str, Any] = Field(default_factory=dict)
    qosDecs: dict[str, Any] = Field(default_factory=dict)
    sessRules: dict[str, Any] = Field(default_factory=dict)
    traffContDecs: dict[str, Any] = Field(default_factory=dict)
    chgDecs: dict[str, Any] = Field(default_factory=dict)
    urspRules: dict[str, Any] = Field(default_factory=dict)
    appCatalog: list[dict[str, Any]] = Field(default_factory=list)
    flowCatalog: list[dict[str, Any]] = Field(default_factory=list)


class AmPolicyContext(BaseModel):
    polAssociationIDGenerator: int | None = None
    associations: dict[str, PcfAmPolicyControlPolicyAssociation] = Field(default_factory=dict)
    allowedSnssais: list[Snssai] = Field(default_factory=list)
    targetSnssais: list[Snssai] = Field(default_factory=list)
    mappingSnssais: list[MappingOfSnssai] = Field(default_factory=list)
    servAreaRes: ServiceAreaRestriction | None = None
    wlServAreaRes: WirelineServiceAreaRestriction | None = None
    rfsp: int | None = None
    smfSelInfo: SmfSelectionData | None = None
    ueAmbr: Ambr | None = None
    ueSliceMbrs: list[dict[str, Any]] = Field(default_factory=list)
    pras: dict[str, PresenceInfo] = Field(default_factory=dict)


class UeContext(BaseModel):
    identity_context: IdentityContext
    access_mobility_context: AccessMobilityContext = Field(default_factory=AccessMobilityContext)
    am_policy_context: AmPolicyContext = Field(default_factory=AmPolicyContext)
    session_policy_context: SessionPolicyContext = Field(default_factory=SessionPolicyContext)
    serving_nf_context: ServingNfContext = Field(default_factory=ServingNfContext)
    mobility_summary: MobilitySummary = Field(default_factory=MobilitySummary)
    udrUri: str | None = None
    afRoutReq: dict[str, Any] | None = None
    aspId: str | None = None
    ratingGroupData: dict[str, list[int]] = Field(default_factory=dict)

    @property
    def supi(self) -> str:
        return self.identity_context.supi

