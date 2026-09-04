from enum import Enum

from pydantic import BaseModel


class PreemptionCapability(str, Enum):
    NOT_PREEMPT = "NOT_PREEMPT"
    MAY_PREEMPT = "MAY_PREEMPT"

class PreemptionVulnerability(str, Enum):
    NOT_PREEMPTABLE = "NOT_PREEMPTABLE"
    PREEMPTABLE = "PREEMPTABLE"

class Arp(BaseModel):
    priorityLevel: int
    preemptCap: PreemptionCapability
    preemptVuln: PreemptionVulnerability