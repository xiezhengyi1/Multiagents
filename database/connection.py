import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

# Default to a local postgres instance if not provided
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/multiagents_db")

try:
    # Ensure client_encoding is utf8 to avoid encoding issues on Windows
    engine = create_engine(
        DATABASE_URL, 
        echo=False, 
        pool_pre_ping=True,
        connect_args={'client_encoding': 'utf8'}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None
    SessionLocal = None
    Base = declarative_base() # Fallback

def get_db():
    """Generator function for dependency injection (e.g., in FastAPI or manually)"""
    if SessionLocal is None:
        raise RuntimeError("Database engine is not initialized. Check DATABASE_URL.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
