import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

RAW_DATABASE_URL = os.getenv("DATABASE_URL")


def to_sqlalchemy_url(database_url: str) -> str:
    normalized = str(database_url or "").strip()
    if normalized.startswith("postgresql+psycopg://"):
        return normalized
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    return normalized


DATABASE_URL = to_sqlalchemy_url(RAW_DATABASE_URL)

Base = declarative_base()


def _redact_database_url(database_url: str) -> str:
    if not database_url:
        return "<unset>"
    url = make_url(database_url)
    return url.render_as_string(hide_password=True)


if not RAW_DATABASE_URL or not str(RAW_DATABASE_URL).strip():
    raise RuntimeError(
        "DATABASE_URL is not configured. Set it in the environment or .env before starting the application."
    )

try:
    # Ensure client_encoding is utf8 to avoid encoding issues on Windows
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        connect_args={"client_encoding": "utf8"},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    redacted_url = _redact_database_url(DATABASE_URL)
    raise RuntimeError(f"Failed to create database engine for {redacted_url}: {e}") from e

def get_db():
    """Generator function for dependency injection (e.g., in FastAPI or manually)"""
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
