from typing import Any, Dict, Iterable, Optional

# 统一字段别名字典（概念 -> 候选字段名）
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "latency_ms": ("latency_ms", "latency", "lat"),
    "packet_loss_rate": ("packet_loss_rate", "loss_rate", "loss_req"),
    "jitter_ms": ("jitter_ms", "jitter", "jitter_req"),
    "throughput_mbps": ("throughput_mbps", "bw_dl", "gbr_dl"),
    "uplink_mbps": ("bw_ul", "gbr_ul", "throughput_ul_mbps"),
}


def pick_first(data: Dict[str, Any], keys: Iterable[str], fallback: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return fallback


def pick_by_concept(data: Dict[str, Any], concept: str, fallback: Any = None) -> Any:
    keys = FIELD_ALIASES.get(concept, ())
    return pick_first(data, keys, fallback)


def to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)
