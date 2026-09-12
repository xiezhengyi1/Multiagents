from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OpenAPIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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


class Snssai(OpenAPIBaseModel):
    sst: int = Field(
        ...,
        description="Unsigned integer, within the range 0 to 255. Represents the Slice/Service Type.",
    )
    sd: str | None = Field(
        None,
        description="3-octet string, representing the Slice Differentiator, in hexadecimal representation.",
    )


class RouteSelectionParameterSet(OpenAPIBaseModel):
    dnn: str = Field(..., description="String representing a Data Network.")
    snssai: Snssai | None = None
    precedence: int = Field(
        ...,
        description="Unsigned Integer. Sets of parameters that may be used to guide the Route Selection Descriptors of the URSP.",
    )


class EthFlowDescription(OpenAPIBaseModel):
    destMacAddr: str | None = Field(
        None,
        description="String identifying a MAC address formatted in the hexadecimal notation.",
    )
    ethType: str | None = None
    fDesc: str | None = Field(
        None, description="Defines a packet filter of an IP flow."
    )
    fDir: FlowDirection | None = None
    sourceMacAddr: str | None = Field(
        None,
        description="String identifying a MAC address formatted in the hexadecimal notation.",
    )
    vlanTags: list[str] | None = None
    srcMacAddrEnd: str | None = Field(
        None,
        description="String identifying a MAC address formatted in the hexadecimal notation.",
    )
    destMacAddrEnd: str | None = Field(
        None,
        description="String identifying a MAC address formatted in the hexadecimal notation.",
    )


class AppDescriptor(OpenAPIBaseModel):
    osId: str = Field(
        ..., description="Represents the Operating System of the served UE."
    )
    appIds: dict[str, str] = Field(
        ...,
        description="Identifies applications that are running on the UE's operating system.",
    )


class TrafficDescriptorComponents(OpenAPIBaseModel):
    appDescs: dict[str, AppDescriptor] | None = Field(
        None,
        description="Describes the operation systems and the corresponding applications for each operation systems. The key of map is osId.",
    )
    flowDescs: list[str] | None = Field(
        None,
        description="Represents a 3-tuple with protocol, server ip and server port for UL/DL application traffic.",
    )
    domainDescs: list[str] | None = Field(
        None,
        description="FQDN(s) or a regular expression which are used as a domain name matching criteria.",
    )
    ethFlowDescs: list[EthFlowDescription] | None = Field(
        None,
        description="Descriptor(s) for destination information of non-IP traffic in which only ethernet flow description is defined.",
    )
    dnns: list[str] | None = Field(
        None,
        description="This is matched against the DNN information provided by the application.",
    )
    connCaps: list[ConnectionCapabilities] | None = Field(
        None,
        description="This is matched against the information provided by a UE application when it requests a network connection with certain capabilities.",
    )


class UrspRuleRequest(OpenAPIBaseModel):
    trafficDesc: TrafficDescriptorComponents | None = None
    relatPrecedence: int = Field(
        ...,
        description="Unsigned Integer, i.e. only value 0 and integers above 0 are permissible.",
    )
    routeSelParamSets: list[RouteSelectionParameterSet] = Field(
        ...,
        description="Sets of parameters that may be used to guide the Route Selection Descriptors of the URSP.",
    )
