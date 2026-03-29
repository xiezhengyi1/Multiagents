import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/multiagents_db"
RAW_DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def to_sqlalchemy_url(database_url: str) -> str:
    normalized = str(database_url or "").strip()
    if normalized.startswith("postgresql+psycopg://"):
        return normalized
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    return normalized


DATABASE_URL = to_sqlalchemy_url(RAW_DATABASE_URL)

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


__all__ = [
    "Base",
    "DATABASE_URL",
    "RAW_DATABASE_URL",
    "SessionLocal",
    "engine",
    "get_db",
    "to_sqlalchemy_url",
]
