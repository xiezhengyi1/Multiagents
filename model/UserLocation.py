from typing import List, Optional, Dict, Union
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from .RatType import RatType

class TransportProtocol(str, Enum):
    UDP = "UDP"
    TCP = "TCP"

class TnapId(BaseModel):
    ssId: Optional[str] = None
    bssId: Optional[str] = None
    civicAddress: Optional[str] = None

class TwapId(BaseModel):
    ssId: Optional[str] = None
    bssId: Optional[str] = None
    civicAddress: Optional[str] = None

class HfcNodeId(BaseModel):
    hfcNId: Optional[str] = None

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
    nid: Optional[str] = None

class Ecgi(BaseModel):
    plmnId: PlmnId
    eutraCellId: str
    nid: Optional[str] = None

class Ncgi(BaseModel):
    plmnId: PlmnId
    nrCellId: str
    nid: Optional[str] = None

class GNbId(BaseModel):
    bitLength: int
    gNBValue: str

class AccessType(str, Enum):
    _3GPP_ACCESS = "3GPP_ACCESS"
    NON_3GPP_ACCESS = "NON_3GPP_ACCESS"

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

class AdditionalAccessInfo(BaseModel):
    accessType: Optional[List[AccessType]] = None
    ratType: Optional[RatType] = None

class PlmnIdNid(BaseModel):
    mcc: str
    mnc: str
    nid: Optional[str] = None

class EutraLocation(BaseModel):
    tai: Tai
    ignoreTai: Optional[bool] = None
    ecgi: Ecgi
    ignoreEcgi: Optional[bool] = None
    ageOfLocationInformation: Optional[int] = None
    ueLocationTimestamp: Optional[datetime] = None
    geographicalInformation: Optional[str] = None
    geodeticInformation: Optional[str] = None
    globalNgenbId: Optional[GlobalRanNodeId] = None
    globalENbId: Optional[GlobalRanNodeId] = None

class NrLocation(BaseModel):
    tai: Tai
    ncgi: Ncgi
    ignoreNcgi: Optional[bool] = None
    ageOfLocationInformation: Optional[int] = None
    ueLocationTimestamp: Optional[datetime] = None
    geographicalInformation: Optional[str] = None
    geodeticInformation: Optional[str] = None
    globalGnbId: Optional[GlobalRanNodeId] = None

class N3gaLocation(BaseModel):
    n3gppTai: Optional[Tai] = None
    n3IwfId: Optional[str] = None
    ueIpv4Addr: Optional[str] = None
    ueIpv6Addr: Optional[str] = None
    portNumber: Optional[int] = None
    protocol: Optional[TransportProtocol] = None
    tnapId: Optional[TnapId] = None
    twapId: Optional[TwapId] = None
    hfcNodeId: Optional[HfcNodeId] = None
    gli: Optional[str] = None
    w5gbanLineType: Optional[LineType] = None
    gci: Optional[str] = None

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
    cgi: Optional[CellGlobalId] = None
    sai: Optional[ServiceAreaId] = None
    lai: Optional[LocationAreaId] = None
    rai: Optional[RoutingAreaId] = None
    ageOfLocationInformation: Optional[int] = None
    ueLocationTimestamp: Optional[datetime] = None
    geographicalInformation: Optional[str] = None
    geodeticInformation: Optional[str] = None

class GeraLocation(BaseModel):
    locationNumber: Optional[str] = None
    cgi: Optional[CellGlobalId] = None
    sai: Optional[ServiceAreaId] = None
    lai: Optional[LocationAreaId] = None
    rai: Optional[RoutingAreaId] = None
    vlrNumber: Optional[str] = None
    mscNumber: Optional[str] = None
    ageOfLocationInformation: Optional[int] = None
    ueLocationTimestamp: Optional[datetime] = None
    geographicalInformation: Optional[str] = None
    geodeticInformation: Optional[str] = None

class UserLocation(BaseModel):
    eutraLocation: Optional[EutraLocation] = None
    nrLocation: Optional[NrLocation] = None
    n3gaLocation: Optional[N3gaLocation] = None
    utraLocation: Optional[UtraLocation] = None
    geraLocation: Optional[GeraLocation] = None
