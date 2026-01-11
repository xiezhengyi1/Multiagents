from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum

class FlowDirection(str, Enum):
    """List of FlowDirection"""
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    UNSPECIFIED = "UNSPECIFIED"

class ConnectionCapabilities(str, Enum):
    """List of ConnectionCapabilities"""
    IMS = "IMS"
    MMS = "MMS"
    SUPL = "SUPL"
    INTERNET = "INTERNET"

class Snssai(BaseModel):
    sst: int = Field(..., description="Unsigned integer, within the range 0 to 255. Represents the Slice/Service Type.")
    sd: Optional[str] = Field(None, description="3-octet string, representing the Slice Differentiator, in hexadecimal representation.")

class RouteSelectionParameterSet(BaseModel):
    dnn: str = Field(..., description="String representing a Data Network.")
    snssai: Optional[Snssai] = None
    precedence: int = Field(..., description="Unsigned Integer. Sets of parameters that may be used to guide the Route Selection Descriptors of the URSP.")

class EthFlowDescription(BaseModel):
    destMacAddr: Optional[str] = Field(None, description="String identifying a MAC address formatted in the hexadecimal notation.")
    ethType: Optional[str] = None
    fDesc: Optional[str] = Field(None, description="Defines a packet filter of an IP flow.")
    fDir: Optional[FlowDirection] = None
    sourceMacAddr: Optional[str] = Field(None, description="String identifying a MAC address formatted in the hexadecimal notation.")
    vlanTags: Optional[List[str]] = None
    srcMacAddrEnd: Optional[str] = Field(None, description="String identifying a MAC address formatted in the hexadecimal notation.")
    destMacAddrEnd: Optional[str] = Field(None, description="String identifying a MAC address formatted in the hexadecimal notation.")

class AppDescriptor(BaseModel):
    osId: str = Field(..., description="Represents the Operating System of the served UE.")
    appIds: Dict[str, str] = Field(..., description="Identifies applications that are running on the UE's operating system.")

class TrafficDescriptorComponents(BaseModel):
    appDescs: Optional[Dict[str, AppDescriptor]] = Field(None, description="Describes the operation systems and the corresponding applications for each operation systems. The key of map is osId.")
    flowDescs: Optional[List[str]] = Field(None, description="Represents a 3-tuple with protocol, server ip and server port for UL/DL application traffic.")
    domainDescs: Optional[List[str]] = Field(None, description="FQDN(s) or a regular expression which are used as a domain name matching criteria.")
    ethFlowDescs: Optional[List[EthFlowDescription]] = Field(None, description="Descriptor(s) for destination information of non-IP traffic in which only ethernet flow description is defined.")
    dnns: Optional[List[str]] = Field(None, description="This is matched against the DNN information provided by the application.")
    connCaps: Optional[List[ConnectionCapabilities]] = Field(None, description="This is matched against the information provided by a UE application when it requests a network connection with certain capabilities.")

class UrspRuleRequest(BaseModel):
    trafficDesc: Optional[TrafficDescriptorComponents] = None
    relatPrecedence: int = Field(..., description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible.")
    routeSelParamSets: List[RouteSelectionParameterSet] = Field(..., description="Sets of parameters that may be used to guide the Route Selection Descriptors of the URSP.")


