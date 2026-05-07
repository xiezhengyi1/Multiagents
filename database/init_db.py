import sys
from pathlib import Path

from sqlalchemy import text

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
for candidate in (PACKAGE_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from database.connection import engine, Base
from database.langchain_pg import ensure_semantic_knowledge_collection, setup_langgraph_postgres
# Import models so Base metadata is populated
from database.models import (
    AgentArtifact,
    AgentHandoffRecord,
    AgentTask,
    EpisodicExperience,
    GraphEdge,
    GraphMetric,
    GraphNode,
    NetworkGraphSnapshot,
    NetworkStatusSnapshot,
    SemanticKnowledge,
    SessionContext,
    SessionStageResult,
    UeAmPolicyAssociationRecord,
    UeContextRecord,
    UeMobilityEventRecord,
    UeServingNfBindingRecord,
)
from shared.logging import setup_logger

logger = setup_logger(__name__)

def init_db():
    if not engine:
        raise RuntimeError("Database engine not configured.")

    try:
        with engine.connect() as connection:
            logger.info("Enabling pgvector extension...")
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
            
        logger.info("Creating tables...")
        
        Base.metadata.create_all(bind=engine)

        # 关键步骤: 为既有库补齐 network_status_snapshot 拆分列（仅新结构）
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS app_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS slice_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS node_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS mobility_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS policy_data JSONB"))
            connection.execute(text("UPDATE network_status_snapshot SET app_data = '[]'::jsonb WHERE app_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET slice_data = '[]'::jsonb WHERE slice_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET node_data = '[]'::jsonb WHERE node_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET mobility_data = '[]'::jsonb WHERE mobility_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET policy_data = '{}'::jsonb WHERE policy_data IS NULL"))
            connection.execute(text("ALTER TABLE network_status_snapshot DROP COLUMN IF EXISTS snapshot_data"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS app_catalog JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS flow_catalog JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS ursp_rules JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS access_mobility_context JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS am_policy_context JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS serving_nf_context JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS mobility_summary JSONB"))
            connection.execute(text("UPDATE ue_context SET app_catalog = '[]'::jsonb WHERE app_catalog IS NULL"))
            connection.execute(text("UPDATE ue_context SET flow_catalog = '[]'::jsonb WHERE flow_catalog IS NULL"))
            connection.execute(text("UPDATE ue_context SET ursp_rules = '{}'::jsonb WHERE ursp_rules IS NULL"))
            connection.execute(text("UPDATE ue_context SET access_mobility_context = '{}'::jsonb WHERE access_mobility_context IS NULL"))
            connection.execute(text("UPDATE ue_context SET am_policy_context = '{}'::jsonb WHERE am_policy_context IS NULL"))
            connection.execute(text("UPDATE ue_context SET serving_nf_context = '{}'::jsonb WHERE serving_nf_context IS NULL"))
            connection.execute(text("UPDATE ue_context SET mobility_summary = '{}'::jsonb WHERE mobility_summary IS NULL"))
            connection.execute(text("ALTER TABLE session_context ADD COLUMN IF NOT EXISTS current_stage VARCHAR"))
            connection.execute(text("ALTER TABLE session_context ADD COLUMN IF NOT EXISTS current_snapshot_id VARCHAR"))
            connection.execute(text("ALTER TABLE session_context ADD COLUMN IF NOT EXISTS current_artifact_id VARCHAR"))
            connection.execute(text("ALTER TABLE session_context ADD COLUMN IF NOT EXISTS round_index INTEGER DEFAULT 0"))
            connection.execute(text("ALTER TABLE session_context ADD COLUMN IF NOT EXISTS last_error TEXT"))
            connection.execute(text("UPDATE session_context SET current_stage = current_step WHERE current_stage IS NULL"))
            connection.commit()

        logger.info("Initializing semantic knowledge PGVector collection...")
        ensure_semantic_knowledge_collection()

        logger.info("Initializing LangGraph postgres backends...")
        setup_langgraph_postgres()

        logger.info("Database tables created successfully.")
        
    except Exception as e:
        logger.error(f"Error initializing database: {repr(e)}")
        raise

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
