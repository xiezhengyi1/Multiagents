import sys
import os

# Ensure the parent directory is in sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sqlalchemy import text
from database.connection import engine, Base
# Import models so Base metadata is populated
from database.models import SessionContext, EpisodicExperience, SemanticKnowledge, NetworkStatusSnapshot, UeContextRecord
from utils.logger import setup_logger

logger = setup_logger(__name__)

def init_db():
    if not engine:
        logger.error("Database engine not configured.")
        return

    try:
        with engine.connect() as connection:
            logger.info("Enabling pgvector extension...")
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
            
        logger.info("Creating tables...")
        
        # Development helper: Explicitly drop semantic_knowledge to apply schema changes (e.g. adding embedding column)
        # In production, use Alembic for migrations.
        # try:
        #     SemanticKnowledge.__table__.drop(bind=engine, checkfirst=True)
        #     logger.info("Dropped existing semantic_knowledge table to apply schema updates.")
        # except Exception as e:
        #     logger.warning(f"Could not drop semantic_knowledge table: {e}")

        Base.metadata.create_all(bind=engine)

        # 关键步骤: 为既有库补齐 network_status_snapshot 拆分列（仅新结构）
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS app_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS slice_data JSONB"))
            connection.execute(text("ALTER TABLE network_status_snapshot ADD COLUMN IF NOT EXISTS node_data JSONB"))
            connection.execute(text("UPDATE network_status_snapshot SET app_data = '[]'::jsonb WHERE app_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET slice_data = '[]'::jsonb WHERE slice_data IS NULL"))
            connection.execute(text("UPDATE network_status_snapshot SET node_data = '[]'::jsonb WHERE node_data IS NULL"))
            connection.execute(text("ALTER TABLE network_status_snapshot DROP COLUMN IF EXISTS snapshot_data"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS app_catalog JSONB"))
            connection.execute(text("ALTER TABLE ue_context ADD COLUMN IF NOT EXISTS flow_catalog JSONB"))
            connection.execute(text("UPDATE ue_context SET app_catalog = '[]'::jsonb WHERE app_catalog IS NULL"))
            connection.execute(text("UPDATE ue_context SET flow_catalog = '[]'::jsonb WHERE flow_catalog IS NULL"))
            connection.commit()

        logger.info("Database tables created successfully.")
        
    except Exception as e:
        # Use repr to avoid UnicodeDecodeError on Windows if system locale is non-utf8
        logger.error(f"Error initializing database: {repr(e)}")

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
