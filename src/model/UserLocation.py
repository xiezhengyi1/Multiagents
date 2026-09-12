from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from .RatType import RatType


class TransportProtocol(str, Enum):
    UDP = "UDP"
    TCP = "TCP"

class TnapId(BaseModel):
    ssId: str | None = None
    bssId: str | None = None
    civicAddress: str | None = None

class TwapId(BaseModel):
    ssId: str | None = None
    bssId: str | None = None
    civicAddress: str | None = None

class HfcNodeId(BaseModel):
    hfcNId: str | None = None

class LineType(str, Enum):
    DSL = "DSL"
    PON = "PON"

class Ambr(BaseModel):
    uplink: str
    downlink: str

class PlmnId(BaseModel):
    mcc: str
    mnc: str

class Tai(BaseModel):
    plmnId: PlmnId
    tac: str
    nid: str | None = None

class Ecgi(BaseModel):
    plmnId: PlmnId
    eutraCellId: str
    nid: str | None = None

class Ncgi(BaseModel):
    plmnId: PlmnId
    nrCellId: str
    nid: str | None = None

class GNbId(BaseModel):
    bitLength: int
    gNBValue: str

class AccessType(str, Enum):
    _3GPP_ACCESS = "3GPP_ACCESS"
    NON_3GPP_ACCESS = "NON_3GPP_ACCESS"

class GlobalRanNodeId(BaseModel):
    plmnId: PlmnId
    n3IwfId: str | None = None
    gNbId: GNbId | None = None
    ngeNbId: str | None = None
    wagfId: str | None = None
    tngfId: str | None = None
    twifId: str | None = None
    nid: str | None = None
    eNbId: str | None = None

class AdditionalAccessInfo(BaseModel):
    accessType: list[AccessType] | None = None
    ratType: RatType | None = None

class PlmnIdNid(BaseModel):
    mcc: str
    mnc: str
    nid: str | None = None

class EutraLocation(BaseModel):
    tai: Tai
    ignoreTai: bool | None = None
    ecgi: Ecgi
    ignoreEcgi: bool | None = None
    ageOfLocationInformation: int | None = None
    ueLocationTimestamp: datetime | None = None
    geographicalInformation: str | None = None
    geodeticInformation: str | None = None
    globalNgenbId: GlobalRanNodeId | None = None
    globalENbId: GlobalRanNodeId | None = None

class NrLocation(BaseModel):
    tai: Tai
    ncgi: Ncgi
    ignoreNcgi: bool | None = None
    ageOfLocationInformation: int | None = None
    ueLocationTimestamp: datetime | None = None
    geographicalInformation: str | None = None
    geodeticInformation: str | None = None
    globalGnbId: GlobalRanNodeId | None = None

class N3gaLocation(BaseModel):
    n3gppTai: Tai | None = None
    n3IwfId: str | None = None
    ueIpv4Addr: str | None = None
    ueIpv6Addr: str | None = None
    portNumber: int | None = None
    protocol: TransportProtocol | None = None
    tnapId: TnapId | None = None
    twapId: TwapId | None = None
    hfcNodeId: HfcNodeId | None = None
    gli: str | None = None
    w5gbanLineType: LineType | None = None
    gci: str | None = None

class CellGlobalId(BaseModel):
    plmnid: PlmnId
    lac: str
    cellId: str

class ServiceAreaId(BaseModel):
    plmnid: PlmnId
    lac: str
    sac: str

class LocationAreaId(BaseModel):
    plmnid: PlmnId
    lac: str

class RoutingAreaId(BaseModel):
    plmnid: PlmnId
    lac: str
    rac: str

class UtraLocation(BaseModel):
    cgi: CellGlobalId | None = None
    sai: ServiceAreaId | None = None
    lai: LocationAreaId | None = None
    rai: RoutingAreaId | None = None
    ageOfLocationInformation: int | None = None
    ueLocationTimestamp: datetime | None = None
    geographicalInformation: str | None = None
    geodeticInformation: str | None = None

class GeraLocation(BaseModel):
    locationNumber: str | None = None
    cgi: CellGlobalId | None = None
    sai: ServiceAreaId | None = None
    lai: LocationAreaId | None = None
    rai: RoutingAreaId | None = None
    vlrNumber: str | None = None
    mscNumber: str | None = None
    ageOfLocationInformation: int | None = None
    ueLocationTimestamp: datetime | None = None
    geographicalInformation: str | None = None
    geodeticInformation: str | None = None

class UserLocation(BaseModel):
    eutraLocation: EutraLocation | None = None
    nrLocation: NrLocation | None = None
    n3gaLocation: N3gaLocation | None = None
    utraLocation: UtraLocation | None = None
    geraLocation: GeraLocation | None = None
