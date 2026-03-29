from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from database.langchain_pg import build_semantic_knowledge_document, get_semantic_knowledge_store
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


def init_knowledge(knowledge_items):
    """Populate the semantic knowledge PGVector collection with default 6G/5G domain data."""
    store = get_semantic_knowledge_store()
    documents = []
    ids = []

    for item in knowledge_items:
        documents.append(
            build_semantic_knowledge_document(
                key=item["key"],
                category=item.get("category"),
                description=item.get("description"),
                value=item.get("value"),
            )
        )
        ids.append(str(item["key"]))

    store.add_documents(documents, ids=ids)
    logger.info("Knowledge collection initialized with %s items.", len(ids))


if __name__ == "__main__":
    knowledge_items = [
        # --- Slice Profiles (鍒囩墖妯℃澘) ---
        {
            "key": "Slice_eMBB_Standard",
            "category": "Slice_Profile",
            "description": "Standard eMBB (Enhanced Mobile Broadband) Slice. Optimized for high data rates, 4K/8K video streaming, and virtual reality (VR). Focuses on maximizing throughput.",
            "value": {
                "sst": 1,
                "sd": "000001",
                "max_throughput_ul": "500Mbps",
                "max_throughput_dl": "2Gbps",
                "latency": "10-20ms",
                "mobility": "High",
            },
        },
        {
            "key": "Slice_URLLC_Ind_Auto",
            "category": "Slice_Profile",
            "description": "URLLC Slice for Industrial Automation. Extremely low latency and high reliability for factory robots, PLC control, and motion synchronization.",
            "value": {
                "sst": 1,
                "sd": "000002",
                "max_throughput_ul": "100Mbps",
                "max_throughput_dl": "100Mbps",
                "isolation_level": "High",
                "availability": "99.9999%",
                "latency": "<5ms",
            },
        },
        {
            "key": "Slice_mMTC_SmartCity",
            "category": "Slice_Profile",
            "description": "mMTC (Massive Machine Type Communications) Slice for Smart Cities. Designed for massive connection density, low power consumption, and small data packets (sensors, meters).",
            "value": {
                "sst": 2,
                "sd": "000003",
                "connection_density": "1,000,000 devices/km2",
                "max_throughput": "1Mbps",
                "energy_efficiency": "High",
            },
        },
        {
            "key": "Slice_V2X_Advanced",
            "category": "Slice_Profile",
            "description": "V2X (Vehicle-to-Everything) Slice. supports autonomous driving, platooning, and sensor sharing. Requires high reliability and low latency with high mobility support.",
            "value": {
                "sst": 3,
                "sd": "000004",
                "latency": "3-10ms",
                "reliability": "99.999%",
                "mobility": "Up to 500km/h",
            },
        },
        {
            "key": "Slice_Compute_AI",
            "category": "Slice_Profile",
            "description": "6G Compute-Native Slice. Integated sensing and computation for AI model training and inference offloading. High uplink for data ingestion.",
            "value": {
                "sst": 4,
                "sd": "000005",
                "type": "Compute-Aware",
                "uplink_priority": "High",
                "compute_guarantee": "Reserved",
                "edge_integration": "Native",
            },
        },
        {
            "key": "QoS_Config_VoNR",
            "category": "QoS_Config",
            "description": "QoS Profile for Voice over New Radio (VoNR). Conversational voice, delay sensitive.",
            "value": {
                "5qi": 1,
                "arp": {"priority_level": 15, "pre_emption_capability": "disabled", "pre_emption_vulnerability": "enabled"},
                "gbr": {"ul": "128kbps", "dl": "128kbps"},
                "mbr": {"ul": "256kbps", "dl": "256kbps"},
            },
        },
        {
            "key": "QoS_Config_CloudGaming",
            "category": "QoS_Config",
            "description": "QoS Profile for Real-time Cloud Gaming. Requires high bandwidth and relatively low latency for interaction.",
            "value": {
                "5qi": 3,
                "arp": {"priority_level": 7, "pre_emption_capability": "enabled", "pre_emption_vulnerability": "disabled"},
                "gbr": {"ul": "5Mbps", "dl": "25Mbps"},
                "mbr": {"ul": "10Mbps", "dl": "50Mbps"},
            },
        },
        {
            "key": "QoS_Config_RemoteSurgery",
            "category": "QoS_Config",
            "description": "Critical Mission QoS for Remote Surgery. Absolute priority, ultra-low latency, zero packet loss tolerance.",
            "value": {
                "5qi": 82,
                "arp": {"priority_level": 1, "pre_emption_capability": "enabled", "pre_emption_vulnerability": "disabled"},
                "gbr": {"ul": "50Mbps", "dl": "50Mbps"},
                "packet_delay_budget": "10ms",
            },
        },
        {
            "key": "QoS_Config_SmartGrid_Diff_Prot",
            "category": "QoS_Config",
            "description": "Smart Grid Differential Protection. Ultra-reliable communication for power grid failing safely.",
            "value": {
                "5qi": 83,
                "arp": {"priority_level": 2, "pre_emption_capability": "disabled", "pre_emption_vulnerability": "disabled"},
                "reliability": "99.9999%",
                "latency": "5ms",
            },
        },
        {
            "key": "QoS_Config_Buffered_Video",
            "category": "QoS_Config",
            "description": "Standard buffered video streaming (YouTube, Netflix, etc.). Non-GBR, high throughput, delay tolerant.",
            "value": {
                "5qi": 6,
                "arp": {"priority_level": 6, "pre_emption_capability": "disabled", "pre_emption_vulnerability": "enabled"},
                "ambr": "Unlimited",
            },
        },
        {
            "key": "5QI_1_Voice",
            "category": "Standard_Def",
            "description": "5QI Value 1. Service Type: Conversational Voice. GBR.",
            "value": {
                "5qi_value": 1,
                "resource_type": "GBR",
                "priority_level": 20,
                "packet_delay_budget": "100ms",
                "packet_error_rate": "10^-2",
                "example_services": "Conversational Voice",
            },
        },
        {
            "key": "5QI_5_IMS",
            "category": "Standard_Def",
            "description": "5QI Value 5. Service Type: IMS Signaling. Non-GBR. High Priority signaling.",
            "value": {
                "5qi_value": 5,
                "resource_type": "Non-GBR",
                "priority_level": 10,
                "packet_delay_budget": "100ms",
                "packet_error_rate": "10^-6",
                "example_services": "IMS Signaling",
            },
        },
        {
            "key": "5QI_9_Default",
            "category": "Standard_Def",
            "description": "5QI Value 9. Service Type: Video (Buffered) and TCP-based traffic. Default Internet traffic.",
            "value": {
                "5qi_value": 9,
                "resource_type": "Non-GBR",
                "priority_level": 90,
                "packet_delay_budget": "300ms",
                "packet_error_rate": "10^-6",
                "example_services": "Bursty Video, Internet",
            },
        },
        {
            "key": "5QI_82_URLLC",
            "category": "Standard_Def",
            "description": "5QI Value 82. Service Type: Discrete Automation (Small Packets). Delay Critical GBR.",
            "value": {
                "5qi_value": 82,
                "resource_type": "Delay Critical GBR",
                "priority_level": 19,
                "packet_delay_budget": "10ms",
                "packet_error_rate": "10^-5",
                "default_averaging_window": "2000ms",
            },
        },
        {
            "key": "5QI_84_Intel_Transport",
            "category": "Standard_Def",
            "description": "5QI Value 84. Service Type: Intelligent Transport Systems. Delay Critical GBR.",
            "value": {
                "5qi_value": 84,
                "resource_type": "Delay Critical GBR",
                "priority_level": 24,
                "packet_delay_budget": "30ms",
                "packet_error_rate": "10^-5",
            },
        },
        {
            "key": "6G_Scenario_Holographic_Comm",
            "category": "6G_Scenario",
            "description": "6G Scene: Holographic Communication. Replicating a person or object in 3D at a remote location in real-time.",
            "value": {
                "throughput": ">1 Tbps",
                "latency": "<1ms (sub-ms)",
                "jitter": "Zero-jitter",
                "sync": "Strict synchronization required",
            },
        },
        {
            "key": "6G_Scenario_Digital_Twin",
            "category": "6G_Scenario",
            "description": "6G Scene: Digital Twin of Physical World. Real-time mapping of physical cities/factories to virtual models.",
            "value": {
                "requirements": "Massive sensing data upload, real-time rendering downlink",
                "latency": "<10ms round-trip",
                "data_volume": "Petabytes/day",
            },
        },
        {
            "key": "6G_Scenario_Tactile_Internet",
            "category": "6G_Scenario",
            "description": "6G Scene: Tactile Internet. Transmitting touch and actuation over the network (Haptic Codecs).",
            "value": {
                "latency": "<1ms",
                "reliability": "99.99999%",
                "control_loop": "1kHz update rate",
            },
        },
        {
            "key": "6G_Scenario_NTN",
            "category": "6G_Scenario",
            "description": "6G Scene: Non-Terrestrial Networks (Satellite/UAV Integration). Global coverage including oceans and deserts.",
            "value": {
                "topology": "LEO/GEO Satellites + HAPS",
                "challenge": "High propagation delay, Doppler shift",
                "handover": "Frequent satellite handover",
            },
        },
        {
            "key": "6G_Scenario_Semantic_Comm",
            "category": "6G_Scenario",
            "description": "6G Scene: Semantic Communication. Transmitting meaning rather than bits. AI-native coding.",
            "value": {
                "efficiency": "10x-100x improvement over Shannon capacity",
                "architecture": "AI-Encoder/Decoder pair",
                "metric": "Semantic Error Rate (SER)",
            },
        },
    ]
    init_knowledge(knowledge_items)
