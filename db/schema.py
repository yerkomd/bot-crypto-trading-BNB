import logging
from pathlib import Path

from .connection import PostgresDatabase


logger = logging.getLogger(__name__)


def ensure_schema(db: PostgresDatabase) -> None:
    """Ensures `trading` schema and tables exist (idempotent)."""
    schema_sql_path = Path(__file__).resolve().parent / "schema.sql"
    sql = schema_sql_path.read_text(encoding="utf-8")

    def _apply(cur):
        cur.execute(sql)
        return True

    try:
        _ = db.run(_apply, retries=3, swallow=False)
        logger.info("PostgreSQL schema ensured (trading.*)")
    except Exception:
        logger.exception("Failed to ensure PostgreSQL schema")
        raise
