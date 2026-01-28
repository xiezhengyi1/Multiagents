import logging
import sys
import os

# Ensure the parent directory is in sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sqlalchemy import text
from database.connection import engine, Base
# Import models so Base metadata is populated
from database.models import SessionContext, EpisodicExperience, SemanticKnowledge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
        
    except Exception as e:
        # Use repr to avoid UnicodeDecodeError on Windows if system locale is non-utf8
        logger.error(f"Error initializing database: {repr(e)}")

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
