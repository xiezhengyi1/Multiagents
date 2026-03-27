from typing import Dict, Optional

# 单一KPI字典源
KPI_ID_TO_NAME: Dict[int, str] = {
    1: "latency",
    2: "throughput",
    3: "reliability",
    4: "connection_density",
}

KPI_NAME_ALIASES: Dict[str, int] = {
    "latency": 1,
    "lat": 1,
    "delay": 1,
    "throughput": 2,
    "thr": 2,
    "rate": 2,
    "reliability": 3,
    "rel": 3,
    "packet_loss": 3,
    "loss": 3,
    "connection_density": 4,
    "density": 4,
    "conn": 4,
    "connections": 4,
}

KPI_RAN_OPERATOR: Dict[str, str] = {
    "latency": "at_most",
    "throughput": "at_least",
    "reliability": "at_least",
    "connection_density": "at_least",
}

KPI_RAN_UNIT: Dict[str, str] = {
    "latency": "ms",
    "throughput": "Mbps",
    "reliability": "rate",
    "connection_density": "per_km2",
}


def name_to_kpi_id(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    return KPI_NAME_ALIASES.get(name.lower())
