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
        # Migrations (idempotent): widen volume columns for market_klines.
        cur.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'trading'
                      AND table_name = 'market_klines'
                      AND column_name = 'volume'
                ) THEN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'trading'
                          AND table_name = 'market_klines'
                          AND column_name = 'volume'
                          AND (numeric_precision IS DISTINCT FROM 28 OR numeric_scale IS DISTINCT FROM 8)
                    ) THEN
                        ALTER TABLE trading.market_klines ALTER COLUMN volume TYPE NUMERIC(28,8);
                    END IF;
                END IF;

                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'trading'
                      AND table_name = 'market_klines'
                      AND column_name = 'quote_volume'
                ) THEN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'trading'
                          AND table_name = 'market_klines'
                          AND column_name = 'quote_volume'
                          AND (numeric_precision IS DISTINCT FROM 28 OR numeric_scale IS DISTINCT FROM 8)
                    ) THEN
                        ALTER TABLE trading.market_klines ALTER COLUMN quote_volume TYPE NUMERIC(28,8);
                    END IF;
                END IF;
            END $$;
            """
        )
        return True

    try:
        _ = db.run(_apply, retries=3, swallow=False)
        logger.info("PostgreSQL schema ensured (trading.*)")
    except Exception:
        logger.exception("Failed to ensure PostgreSQL schema")
        raise


def ensure_market_schema(db: PostgresDatabase) -> None:
    """Ensures ONLY the market-data schema exists (idempotent).

    This is intended for the Market Data Process (NO trading).
    It avoids touching trading tables like `open_positions`, which may be owned by another role.
    """
    schema_sql_path = Path(__file__).resolve().parent / "schema_market.sql"
    sql = schema_sql_path.read_text(encoding="utf-8")

    def _apply(cur):
        cur.execute(sql)
        # Migrations (idempotent): widen volume columns for market_klines.
        # This prevents numeric overflow on high timeframes (e.g. 1w) for large-volume symbols.
        cur.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'trading'
                      AND table_name = 'market_klines'
                      AND column_name = 'volume'
                ) THEN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'trading'
                          AND table_name = 'market_klines'
                          AND column_name = 'volume'
                          AND (numeric_precision IS DISTINCT FROM 28 OR numeric_scale IS DISTINCT FROM 8)
                    ) THEN
                        ALTER TABLE trading.market_klines ALTER COLUMN volume TYPE NUMERIC(28,8);
                    END IF;
                END IF;

                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'trading'
                      AND table_name = 'market_klines'
                      AND column_name = 'quote_volume'
                ) THEN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'trading'
                          AND table_name = 'market_klines'
                          AND column_name = 'quote_volume'
                          AND (numeric_precision IS DISTINCT FROM 28 OR numeric_scale IS DISTINCT FROM 8)
                    ) THEN
                        ALTER TABLE trading.market_klines ALTER COLUMN quote_volume TYPE NUMERIC(28,8);
                    END IF;
                END IF;
            END $$;
            """
        )
        return True

    try:
        _ = db.run(_apply, retries=3, swallow=False)
        logger.info("PostgreSQL market schema ensured (trading.market_klines)")
    except Exception:
        logger.exception("Failed to ensure PostgreSQL market schema")
        raise
