import sys
import os
import json
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Add project root to python path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import engine
from database.models import SemanticKnowledge, Base
# Import NaturalLanguageEncoder from Embedding.py (ensure consistency)
from multi_agents.Embedding import NaturalLanguageEncoder

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Session = sessionmaker(bind=engine)

def init_knowledge(knowledge_items):
    """Populate Semantic Knowledge with default 6G/5G domain data"""
    
    # Ensure environment variables for DashScope if not already set, 
    # to support NaturalLanguageEncoder which relies on OpenAI client or specific env vars.
    if not os.getenv("OPENAI_BASE_URL"):
         # Default DashScope compatible endpoint
         os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # If DASHSCOPE_API_KEY is present but OPENAI_API_KEY is not, set OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY") and os.getenv("DASHSCOPE_API_KEY"):
         os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

    session = Session()
    try:
        # Use NaturalLanguageEncoder from Embedding.py
        encoder = NaturalLanguageEncoder()
        logger.info(f"Initialized NaturalLanguageEncoder with model: {encoder.embedding_model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize encoder: {e}")
        return

    # Combined knowledge data from all sources

    for item in knowledge_items:
        # Create text for embedding (combine key, category, description, and potentially value content)
        text_to_embed = f"Key: {item['key']}. Category: {item['category']}. Description: {item['description']}"
        # Adding value summary can help if semantic search queries detailed specs
        if isinstance(item.get('value'), dict):
             # Convert simple dict to string representation for embedding context
             val_str = ", ".join([f"{k}: {v}" for k,v in item['value'].items() if isinstance(v, (str, int, float))])
             text_to_embed += f". Details: {val_str}"

        logger.info(f"Processing item: {item['key']}")
        
        try:
            # Use NaturalLanguageEncoder to get vector
            # encode returns: {"raw_text": ..., "detected_keywords": ..., "vector": ...}
            result = encoder.encode(text_to_embed)
            vector = result.get("vector")
            
            if not vector:
                logger.warning(f"No embedding vector generated for {item['key']}")
                continue

            # Upsert logic
            existing = session.query(SemanticKnowledge).filter_by(key=item['key']).first()
            if existing:
                existing.category = item['category']
                existing.description = item['description']
                existing.value = item['value']
                existing.embedding = vector            
                existing.updated_at = datetime.utcnow()
                logger.info(f"Updated {item['key']}")
            else:
                new_entry = SemanticKnowledge(
                    key=item['key'],
                    category=item['category'],
                    description=item['description'],
                    value=item['value'],
                    embedding=vector
                )
                session.add(new_entry)
                logger.info(f"Inserted {item['key']}")
                
        except Exception as e:
            logger.error(f"Failed to process {item['key']}: {e}")
            continue

    try:
        session.commit()
        logger.info("Knowledge Base initialization complete.")
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    knowledge_items = [
        # 1. Slice Profile for Industrial Automation
        {
            "key": "Slice_Ind_Auto_Legacy",
            "category": "Slice_Profile", 
            "description": "Slice Profile optimized for Industrial Automation, Mechanical Arm Control, Factory Automation, Robotics, and Manufacturing lines. Requires extremely low latency (URLLC), high reliability, and precise synchronization for motion control. Supports jitter-sensitive applications. Keywords: industrial robot, plc control, smart factory.",
            "value": {
                "sst": 1, 
                "sd": "000002", 
                "max_throughput_ul": "100Mbps",
                "max_throughput_dl": "100Mbps",
                "isolation_level": "High",
                "availability": "99.9999%"
            }
        },
        # 2. QoS Config for URLLC
        {
            "key": "QoS_Config_URLLC_Standard",
            "category": "QoS_Config",
            "description": "Standard QoS Configuration for Ultra-Reliable Low Latency Communication (URLLC). Applicable to remote surgery, V2X (Vehicle-to-Everything), autonomous driving, and tactile internet. Prioritizes packet delivery speed and reliability over throughput. Keywords: low delay, high reliability, critical mission.",
            "value": {
                "5qi": 82,
                "arp": {"priority_level": 3, "pre_emption_capability": "enabled", "pre_emption_vulnerability": "disabled"},
                "gbr": {"ul": "10Mbps", "dl": "10Mbps"},
                "mbr": {"ul": "20Mbps", "dl": "20Mbps"}
            }
        },
        # 3. Specific Key: 5qi_urllc
        {
            "key": "5qi_urllc",
            "category": "Standard_Def",
            "description": "Definition of 5QI Value 82 for URLLC services. Critical for delay-sensitive flows. Keywords: 5qi_urllec, 5qi_urlc, ultra reliable low latency.",
            "value": {
                "5qi_value": 82,
                "resource_type": "Delay Critical GBR",
                "priority_level": 19,
                "packet_delay_budget": "10ms",
                "packet_error_rate": "10^-5",
                "default_averaging_window": "2000ms"
            }
        },
        # 4. Immersive XR
        {
            "key": "6G_Scenario_Immersive_XR",
            "category": "6G_Scenario",
            "description": "6G Scene: Immersive cloud XR services. High-fidelity extended reality applications requiring ultra-low latency and high bandwidth.",
            "value": {
                "name": "Immersive XR (Extended Reality)",
                "latency_requirement": "< 5ms",
                "throughput_requirement": "> 1 Gbps",
                "reliability": "99.999%"
            }
        },
        # 5. Holographic Comm
        {
            "key": "6G_Scenario_Holographic_Comm",
            "category": "6G_Scenario",
            "description": "6G Scene: High-fidelity holographic type communications. Wait-free, full-dimensional holographic communication.",
            "value": {
                "name": "Holographic Communication",
                "latency_requirement": "< 1ms",
                "throughput_requirement": "> 1 Tbps (peak)",
                "reliability": "99.99999%"
            }
        },
        # 6. Autonomous Driving
        {
            "key": "6G_Scenario_V2X_Autonomous",
            "category": "6G_Scenario",
            "description": "6G Scene: Advanced autonomous driving and vehicular networks. Fully autonomous driving coordinated via vehicle-to-everything communication.",
            "value": {
                "name": "Autonomous Driving (V2X)",
                "latency_requirement": "< 3ms",
                "reliability": "99.99999%",
                "mobility": "> 500 km/h"
            }
        },
        # 7. Massive IoT
        {
            "key": "6G_Scenario_Massive_IoT",
            "category": "6G_Scenario",
            "description": "6G Scene: Massive Internet of Things and Digital Twins. Connecting billions of sensors and devices for digital twins of smart cities.",
            "value": {
                "name": "Massive Digital Twin / IoT",
                "connection_density": "10^7 devices/km2",
                "energy_efficiency": "Ultra-high"
            }
        },
        # 8. Tactile Internet
        {
            "key": "6G_Scenario_Tactile_Internet",
            "category": "6G_Scenario",
            "description": "6G Scene: Tactile Internet for remote surgery and industry. Remote physical interaction with haptic feedback.",
            "value": {
                "name": "Tactile Internet / Telepresence",
                "latency_requirement": "< 1ms",
                "jitter": "< 10us"
            }
        }
    ]
    init_knowledge(knowledge_items)