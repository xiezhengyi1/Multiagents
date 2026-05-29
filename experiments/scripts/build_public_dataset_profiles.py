from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any
import ipaddress

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "experiments" / "public_datasets" / "raw"
DERIVED_DIR = ROOT / "experiments" / "public_datasets" / "derived"
SCENARIO_DIR = ROOT / "experiments" / "scenarios"
OUTPUT_SCENARIO_DIR = ROOT / "experiments" / "scenarios_public"


DATASET_SOURCES = {
    "im_video_calls": {
        "title": "Instant Messaging Meets Video Conferencing: Studying the Performance of IM Video Calls",
        "doi": "10.5281/zenodo.8006901",
        "url": "https://zenodo.org/records/8006901",
    },
    "https_traffic": {
        "title": "Dataset used for HTTPS traffic classification using packet burst statistics",
        "doi": "10.5281/zenodo.4911551",
        "url": "https://zenodo.org/records/4911551",
    },
    "cloud_gaming": {
        "title": "VR-AR-CG Network Traffic Datasets",
        "url": "https://github.com/dcomp-leris/VR-AR-CG-network-telemetry",
    },
    "fanet": {
        "title": "FANET Dataset: UAV Communication Scenarios in NS-3.40",
        "doi": "10.5281/zenodo.19373220",
        "url": "https://zenodo.org/records/19373220",
    },
    "lorawan": {
        "title": "LoRaWAN Traffic Analysis Dataset",
        "doi": "10.5281/zenodo.8090619",
        "url": "https://zenodo.org/records/8090619",
    },
}


FLOW_PROFILE_MAPPING = {
    "Telemedicine_video_1": "video_conference_proxy",
    "Remote_Drive_video_1": "video_conference_proxy",
    "Factory_Robot_video_1": "video_conference_proxy",
    "4K_Video_stream_1": "video_player_stream",
    "4K_Video_control_2": "cloud_gaming_control",
    "AR_Gaming_video_2": "cloud_gaming_stream",
    "AR_Gaming_control_1": "cloud_gaming_control",
    "Cloud_Render_stream_1": "cloud_gaming_stream",
    "Cloud_Render_telemetry_2": "cloud_gaming_control",
    "Drone_Control_video_1": "uav_control_proxy",
    "IoT_Sensor_telemetry_1": "iot_lorawan_proxy",
    "IoT_Sensor_control_2": "iot_lorawan_proxy",
    "Smart_Meter_telemetry_1": "iot_lorawan_proxy",
    "Web_Browse_session_1": "web_browsing_session",
    "Web_Browse_session_2": "web_browsing_session",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset artifact: {path}")
    return path


def _safe_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mbps_to_pps(mbps: float, packet_size_bytes: float) -> float:
    if mbps <= 0 or packet_size_bytes <= 0:
        return 0.0
    return (mbps * 1_000_000.0) / (8.0 * packet_size_bytes)


def _weighted_packet_size(dl_size: float, ul_size: float, dl_pps: float, ul_pps: float) -> float:
    total = dl_pps + ul_pps
    if total <= 0:
        return max(dl_size, ul_size)
    return ((dl_size * dl_pps) + (ul_size * ul_pps)) / total


def _preserve_guarantee(old_bandwidth: float, old_guaranteed: float, new_bandwidth: float) -> float:
    if old_bandwidth <= 0:
        return 0.0
    return new_bandwidth * (old_guaranteed / old_bandwidth)


def compute_video_conference_proxy() -> dict[str, Any]:
    zip_path = _require(RAW_DIR / "im_video_calls.zip")
    with zipfile.ZipFile(zip_path) as zf:
        index_path = next(name for name in zf.namelist() if name.endswith("index.csv"))
        avg_path = next(name for name in zf.namelist() if name.endswith("index_avgbwdist.csv"))
        index_df = pd.read_csv(zf.open(index_path))
        avg_df = pd.read_csv(zf.open(avg_path))

    df = index_df.merge(avg_df, on="guid")
    df["call_mbps"] = df["call_bytes"] * 8.0 / df["obsdur"] / 1_000_000.0

    total_mbps = float(df["call_mbps"].median())
    dl_mbps = total_mbps / 2.0
    ul_mbps = total_mbps / 2.0
    packet_size = 1000.0
    qdelay_ms = float(df["qdelay"].median() * 1000.0)
    qdelay_iqr_ms = float((df["qdelay"].quantile(0.75) - df["qdelay"].quantile(0.25)) * 1000.0)

    return {
        "profile_id": "video_conference_proxy",
        "source_dataset_ids": ["im_video_calls"],
        "mapping_note": "Used as a proxy for bidirectional low-latency visual interaction flows. Packet size remains a packetization assumption because this dataset exposes aggregate call throughput, queueing delay, and loss rather than per-packet payload sizes.",
        "supported_fields": [
            "packet_size_bytes",
            "dl_packet_size_bytes",
            "ul_packet_size_bytes",
            "arrival_rate_pps",
            "dl_arrival_rate_pps",
            "ul_arrival_rate_pps",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
            "sla_target.latency_ms",
            "sla_target.jitter_ms",
            "sla_target.loss_rate",
        ],
        "stats": {
            "dl_packet_size_bytes": packet_size,
            "ul_packet_size_bytes": packet_size,
            "dl_bandwidth_mbps": dl_mbps,
            "ul_bandwidth_mbps": ul_mbps,
            "dl_arrival_rate_pps": _mbps_to_pps(dl_mbps, packet_size),
            "ul_arrival_rate_pps": _mbps_to_pps(ul_mbps, packet_size),
            "latency_ms": qdelay_ms,
            "jitter_ms": qdelay_iqr_ms,
            "loss_rate": float(df["callDropRate"].median()),
            "derivation": {
                "call_mbps_median": total_mbps,
                "qdelay_ms_median": qdelay_ms,
                "qdelay_ms_iqr": qdelay_iqr_ms,
                "packet_size_assumption_bytes": packet_size,
            },
        },
    }


def compute_https_profiles() -> tuple[dict[str, Any], dict[str, Any]]:
    csv_path = _require(RAW_DIR / "https_clf_dataset.csv")
    cols = [
        "BYTES",
        "BYTES_REV",
        "TYPE",
        "PKT_LENGTHS_MEAN",
        "BRST_DURATION_MEAN",
        "BRST_INTERVALS_MEAN",
        "BRST_COUNT",
    ]
    df = pd.read_csv(csv_path, usecols=cols)

    def profile_for(label: str, quantile: float, profile_id: str, note: str) -> dict[str, Any]:
        sub = df[df["TYPE"] == label].copy()
        dl_bytes = sub[["BYTES", "BYTES_REV"]].max(axis=1)
        ul_bytes = sub[["BYTES", "BYTES_REV"]].min(axis=1)
        duration = (
            sub["BRST_DURATION_MEAN"] * sub["BRST_COUNT"]
            + sub["BRST_INTERVALS_MEAN"] * (sub["BRST_COUNT"] - 1).clip(lower=0)
        )
        dl_mbps = (dl_bytes * 8.0 / duration / 1_000_000.0).quantile(quantile)
        ul_mbps = (ul_bytes * 8.0 / duration / 1_000_000.0).quantile(quantile)
        packet_size = sub["PKT_LENGTHS_MEAN"].quantile(quantile)
        return {
            "profile_id": profile_id,
            "source_dataset_ids": ["https_traffic"],
            "mapping_note": note,
            "supported_fields": [
                "packet_size_bytes",
                "dl_packet_size_bytes",
                "ul_packet_size_bytes",
                "arrival_rate_pps",
                "dl_arrival_rate_pps",
                "ul_arrival_rate_pps",
                "allocated_bandwidth_dl_mbps",
                "allocated_bandwidth_ul_mbps",
            ],
            "stats": {
                "dl_packet_size_bytes": float(packet_size),
                "ul_packet_size_bytes": float(packet_size),
                "dl_bandwidth_mbps": float(dl_mbps),
                "ul_bandwidth_mbps": float(ul_mbps),
                "dl_arrival_rate_pps": _mbps_to_pps(float(dl_mbps), float(packet_size)),
                "ul_arrival_rate_pps": _mbps_to_pps(float(ul_mbps), float(packet_size)),
                "derivation": {
                    "quantile": quantile,
                    "traffic_label": label,
                },
            },
        }

    video_player = profile_for(
        label="P",
        quantile=0.5,
        profile_id="video_player_stream",
        note="Applied to high-rate downlink media flows such as 4K video. The dataset category is HTTPS video player traffic.",
    )
    web_browsing = profile_for(
        label="W",
        quantile=0.85,
        profile_id="web_browsing_session",
        note="Applied to website-style flows. The 0.85 quantile is used to avoid collapsing the profile to the extremely small short-session median.",
    )
    return video_player, web_browsing


def compute_cloud_gaming_profiles() -> tuple[dict[str, Any], dict[str, Any]]:
    csv_files = sorted(RAW_DIR.glob("cg_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No cloud gaming CSV features found under experiments/public_datasets/raw")

    def is_private_ip(value: str) -> bool:
        try:
            return ipaddress.ip_address(str(value)).is_private
        except ValueError:
            return False

    uplink_rows: list[dict[str, float]] = []
    downlink_rows: list[dict[str, float]] = []

    for path in csv_files:
        df = pd.read_csv(path)
        df = df[df["Protocol"] == "UDP"].copy()
        df["src_private"] = df["SrcIP"].map(is_private_ip)
        df["dst_private"] = df["DstIP"].map(is_private_ip)

        uplink = df[df["src_private"] & ~df["dst_private"]].sort_values("FlowSizeBytes", ascending=False).head(1)
        downlink = df[~df["src_private"] & df["dst_private"]].sort_values("FlowSizeBytes", ascending=False).head(1)
        if not uplink.empty:
            uplink_rows.append(uplink[["PS", "IPI", "FlowSizeBytes"]].iloc[0].to_dict())
        if not downlink.empty:
            downlink_rows.append(downlink[["PS", "IPI", "FlowSizeBytes"]].iloc[0].to_dict())

    up_df = pd.DataFrame(uplink_rows)
    down_df = pd.DataFrame(downlink_rows)

    control_packet_size = float(up_df["PS"].median())
    control_bandwidth = float((up_df["PS"] * 8.0 / up_df["IPI"] / 1_000_000.0).median())
    stream_packet_size = float(down_df["PS"].median())
    stream_bandwidth = float((down_df["PS"] * 8.0 / down_df["IPI"] / 1_000_000.0).median())

    control_profile = {
        "profile_id": "cloud_gaming_control",
        "source_dataset_ids": ["cloud_gaming"],
        "mapping_note": "Applied to interactive control and telemetry flows. The dominant client-to-server UDP flow from each 5G cloud-gaming sample is used as the control proxy.",
        "supported_fields": [
            "packet_size_bytes",
            "dl_packet_size_bytes",
            "ul_packet_size_bytes",
            "arrival_rate_pps",
            "dl_arrival_rate_pps",
            "ul_arrival_rate_pps",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
        ],
        "stats": {
            "dl_packet_size_bytes": control_packet_size,
            "ul_packet_size_bytes": control_packet_size,
            "dl_bandwidth_mbps": control_bandwidth * 0.2,
            "ul_bandwidth_mbps": control_bandwidth,
            "dl_arrival_rate_pps": _mbps_to_pps(control_bandwidth * 0.2, control_packet_size),
            "ul_arrival_rate_pps": _mbps_to_pps(control_bandwidth, control_packet_size),
            "derivation": {
                "aggregation": "median dominant uplink UDP flow across 5G cloud gaming feature files",
                "ack_downlink_ratio": 0.2,
            },
        },
    }

    stream_profile = {
        "profile_id": "cloud_gaming_stream",
        "source_dataset_ids": ["cloud_gaming"],
        "mapping_note": "Applied to low-latency rendered-media downlink flows. The dominant server-to-client UDP flow from each 5G cloud-gaming sample is used as the streaming proxy.",
        "supported_fields": [
            "packet_size_bytes",
            "dl_packet_size_bytes",
            "ul_packet_size_bytes",
            "arrival_rate_pps",
            "dl_arrival_rate_pps",
            "ul_arrival_rate_pps",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
        ],
        "stats": {
            "dl_packet_size_bytes": stream_packet_size,
            "ul_packet_size_bytes": control_packet_size,
            "dl_bandwidth_mbps": stream_bandwidth,
            "ul_bandwidth_mbps": control_bandwidth,
            "dl_arrival_rate_pps": _mbps_to_pps(stream_bandwidth, stream_packet_size),
            "ul_arrival_rate_pps": _mbps_to_pps(control_bandwidth, control_packet_size),
            "derivation": {
                "aggregation": "median dominant downlink UDP flow across 5G cloud gaming feature files",
            },
        },
    }
    return control_profile, stream_profile


def compute_uav_control_proxy() -> dict[str, Any]:
    zip_path = _require(RAW_DIR / "fanet.zip")
    with zipfile.ZipFile(zip_path) as zf:
        packet_trace = pd.read_csv(zf.open("FANET_Dataset_NS3.40/Scenario_1/packet_trace.csv"))
        qos = pd.read_csv(zf.open("FANET_Dataset_NS3.40/Scenario_1/network_qos_metrics.csv"))

    data_df = packet_trace[packet_trace["PacketTypePrimary"] == "DATA"].copy()
    packet_size = float(data_df["PacketSizeByte_tx"].median())
    throughput_mbps = float(qos["throughput_bps"].median() / 1_000_000.0)

    return {
        "profile_id": "uav_control_proxy",
        "source_dataset_ids": ["fanet"],
        "mapping_note": "Applied to UAV-related control/video uplink flows. Scenario 1 is selected because it is the lowest-loss FANET configuration in the downloaded archive.",
        "supported_fields": [
            "packet_size_bytes",
            "dl_packet_size_bytes",
            "ul_packet_size_bytes",
            "arrival_rate_pps",
            "dl_arrival_rate_pps",
            "ul_arrival_rate_pps",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
            "sla_target.latency_ms",
            "sla_target.jitter_ms",
            "sla_target.loss_rate",
        ],
        "stats": {
            "dl_packet_size_bytes": packet_size,
            "ul_packet_size_bytes": packet_size,
            "dl_bandwidth_mbps": throughput_mbps * 0.1,
            "ul_bandwidth_mbps": throughput_mbps,
            "dl_arrival_rate_pps": _mbps_to_pps(throughput_mbps * 0.1, packet_size),
            "ul_arrival_rate_pps": _mbps_to_pps(throughput_mbps, packet_size),
            "latency_ms": float(qos["avg_delay_ms"].median()),
            "jitter_ms": float(qos["jitter_ms"].median()),
            "loss_rate": float(qos["LossRate"].median()),
            "derivation": {
                "scenario_id": 1,
                "downlink_ack_ratio": 0.1,
            },
        },
    }


def compute_iot_lorawan_proxy() -> dict[str, Any]:
    zip_path = _require(RAW_DIR / "lorawan_csv.zip")
    headers = [
        "frame_number",
        "time_epoch",
        "payload_len",
        "srcgw",
        "crc",
        "rssi_dbm",
        "snr_db",
        "frequency_hz",
        "sf",
        "cr",
        "ftype",
        "devaddr",
        "fport",
        "fcnt",
        "flags",
        "airtime_ms",
    ]
    files = ["02_Liege_data.csv", "04_Graz_data.csv", "05_Wien_data.csv", "07_Brno_data.csv"]
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in files:
            frame = pd.read_csv(zf.open(name), names=headers)
            frame["time_epoch"] = pd.to_numeric(frame["time_epoch"], errors="coerce")
            frame["payload_len"] = pd.to_numeric(frame["payload_len"], errors="coerce")
            frame["airtime_ms"] = pd.to_numeric(frame["airtime_ms"], errors="coerce")
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["time_epoch", "payload_len", "airtime_ms"])
    pps = 1.0 / float(df["time_epoch"].sort_values().diff().dropna().median())
    packet_size = float(df["payload_len"].median())
    bandwidth = pps * packet_size * 8.0 / 1_000_000.0

    return {
        "profile_id": "iot_lorawan_proxy",
        "source_dataset_ids": ["lorawan"],
        "mapping_note": "Applied to sparse IoT telemetry flows. The statistics are derived from valid LoRaWAN data packets in the evaluated city captures.",
        "supported_fields": [
            "packet_size_bytes",
            "dl_packet_size_bytes",
            "ul_packet_size_bytes",
            "arrival_rate_pps",
            "dl_arrival_rate_pps",
            "ul_arrival_rate_pps",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
        ],
        "stats": {
            "dl_packet_size_bytes": packet_size,
            "ul_packet_size_bytes": packet_size,
            "dl_bandwidth_mbps": bandwidth * 0.1,
            "ul_bandwidth_mbps": bandwidth,
            "dl_arrival_rate_pps": pps * 0.1,
            "ul_arrival_rate_pps": pps,
            "derivation": {
                "airtime_ms_median": float(df["airtime_ms"].median()),
                "evaluated_files": files,
                "downlink_ack_ratio": 0.1,
            },
        },
    }


def build_profile_catalog() -> dict[str, Any]:
    video_player_stream, web_browsing_session = compute_https_profiles()
    cloud_gaming_control, cloud_gaming_stream = compute_cloud_gaming_profiles()
    profiles = [
        compute_video_conference_proxy(),
        video_player_stream,
        web_browsing_session,
        cloud_gaming_control,
        cloud_gaming_stream,
        compute_uav_control_proxy(),
        compute_iot_lorawan_proxy(),
    ]
    return {
        "dataset_sources": DATASET_SOURCES,
        "profiles": {profile["profile_id"]: profile for profile in profiles},
        "flow_profile_mapping": FLOW_PROFILE_MAPPING,
    }


def apply_profiles_to_scenarios(catalog: dict[str, Any]) -> dict[str, str]:
    profiles = catalog["profiles"]
    OUTPUT_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    for source in sorted(SCENARIO_DIR.glob("*.yaml")):
        with source.open("r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle)

        base_name = scenario["name"]
        base_id = scenario["scenario_id"]
        scenario["name"] = f"{base_name}-public-datasets"
        scenario["scenario_id"] = f"{base_id}-public-datasets"

        if "free5gc" in scenario and "project_name" in scenario["free5gc"]:
            scenario["free5gc"]["project_name"] = f"{scenario['free5gc']['project_name']}-public"

        for flow in scenario.get("flows", []):
            profile_id = FLOW_PROFILE_MAPPING.get(flow["name"])
            if not profile_id:
                continue

            profile = profiles[profile_id]["stats"]
            flow["dl_packet_size_bytes"] = _safe_round(profile["dl_packet_size_bytes"], 3)
            flow["ul_packet_size_bytes"] = _safe_round(profile["ul_packet_size_bytes"], 3)
            flow["dl_arrival_rate_pps"] = _safe_round(profile["dl_arrival_rate_pps"], 3)
            flow["ul_arrival_rate_pps"] = _safe_round(profile["ul_arrival_rate_pps"], 3)
            flow["allocated_bandwidth_dl_mbps"] = _safe_round(profile["dl_bandwidth_mbps"], 6)
            flow["allocated_bandwidth_ul_mbps"] = _safe_round(profile["ul_bandwidth_mbps"], 6)
            flow["packet_size_bytes"] = _safe_round(
                _weighted_packet_size(
                    profile["dl_packet_size_bytes"],
                    profile["ul_packet_size_bytes"],
                    profile["dl_arrival_rate_pps"],
                    profile["ul_arrival_rate_pps"],
                ),
                3,
            )
            flow["arrival_rate_pps"] = _safe_round(
                profile["dl_arrival_rate_pps"] + profile["ul_arrival_rate_pps"],
                3,
            )

            sla = flow.setdefault("sla_target", {})
            old_dl_bw = float(sla.get("bandwidth_dl_mbps", flow.get("allocated_bandwidth_dl_mbps", 0.0)))
            old_ul_bw = float(sla.get("bandwidth_ul_mbps", flow.get("allocated_bandwidth_ul_mbps", 0.0)))
            old_g_dl = float(sla.get("guaranteed_bandwidth_dl_mbps", 0.0))
            old_g_ul = float(sla.get("guaranteed_bandwidth_ul_mbps", 0.0))

            sla["bandwidth_dl_mbps"] = _safe_round(profile["dl_bandwidth_mbps"], 6)
            sla["bandwidth_ul_mbps"] = _safe_round(profile["ul_bandwidth_mbps"], 6)
            sla["guaranteed_bandwidth_dl_mbps"] = _safe_round(
                _preserve_guarantee(old_dl_bw, old_g_dl, profile["dl_bandwidth_mbps"]),
                6,
            )
            sla["guaranteed_bandwidth_ul_mbps"] = _safe_round(
                _preserve_guarantee(old_ul_bw, old_g_ul, profile["ul_bandwidth_mbps"]),
                6,
            )

            if "latency_ms" in profile:
                sla["latency_ms"] = _safe_round(profile["latency_ms"], 6)
            if "jitter_ms" in profile:
                sla["jitter_ms"] = _safe_round(profile["jitter_ms"], 6)
            if "loss_rate" in profile:
                sla["loss_rate"] = _safe_round(profile["loss_rate"], 9)

        output_path = OUTPUT_SCENARIO_DIR / f"{source.stem}_public_datasets.yaml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(scenario, handle, allow_unicode=False, sort_keys=False)
        outputs[source.name] = str(output_path.relative_to(ROOT))

    return outputs


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_profile_catalog()
    scenario_outputs = apply_profiles_to_scenarios(catalog)
    catalog["generated_scenarios"] = scenario_outputs

    derived_path = DERIVED_DIR / "profile_catalog.json"
    with derived_path.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "profile_catalog": str(derived_path.relative_to(ROOT)),
        "generated_scenarios": scenario_outputs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
