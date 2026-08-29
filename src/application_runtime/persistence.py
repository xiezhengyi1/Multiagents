"""Application composition for the shared runtime persistence port.

Database imports stay inside the factory so importing an orchestrator or agent
does not initialize database configuration as a side effect.
"""

from __future__ import annotations

from shared.runtime.storage.persistence import RuntimePersistence
from shared.runtime.storage.runtime_store import SqlAlchemyRuntimePersistence


def create_runtime_persistence() -> RuntimePersistence:
    """Build a SQLAlchemy-backed runtime persistence instance for one app."""

    from database.connection import SessionLocal
    import database.models as models

    return SqlAlchemyRuntimePersistence(session_factory=SessionLocal, orm_module=models)


__all__ = ["create_runtime_persistence"]
