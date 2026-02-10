import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from db.connection import PostgresDatabase


logger = logging.getLogger(__name__)


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None


def _to_dt_naive_utc(v: Any) -> datetime:
    """Converts datetime-like to naive UTC datetime (TIMESTAMP without time zone)."""
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            return v.astimezone(timezone.utc).replace(tzinfo=None)
        return v
    try:
        dt = v.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        raise TypeError(f"Unsupported datetime type: {type(v)}")


class MarketKlinesRepository:
    def __init__(self, db: PostgresDatabase):
        self._db = db

    def last_open_time(self, *, symbol: str, interval: str) -> Optional[datetime]:
        symbol = str(symbol).upper().strip()
        interval = str(interval).strip()

        def _q(cur):
            cur.execute(
                """
                SELECT open_time
                FROM trading.market_klines
                WHERE symbol = %s AND interval = %s
                ORDER BY open_time DESC
                LIMIT 1
                """,
                (symbol, interval),
            )
            row = cur.fetchone()
            return row.get("open_time") if row else None

        dt = self._db.run(_q, retries=2, swallow=True)
        if dt is None:
            return None
        if isinstance(dt, datetime):
            # stored as TIMESTAMP (naive UTC)
            return dt.replace(tzinfo=None)
        try:
            return _to_dt_naive_utc(dt)
        except Exception:
            return None

    def insert_many(self, *, symbol: str, interval: str, rows: list[dict]) -> int:
        """Bulk insert rows with UPSERT semantics (ON CONFLICT DO NOTHING)."""
        symbol = str(symbol).upper().strip()
        interval = str(interval).strip()
        if not rows:
            return 0

        def _q(cur):
            try:
                from psycopg2.extras import execute_values
            except Exception as e:
                raise RuntimeError("psycopg2 extras missing; install psycopg2-binary") from e

            values = []
            for r in rows:
                values.append(
                    (
                        symbol,
                        interval,
                        _to_dt_naive_utc(r["open_time"]),
                        _to_dt_naive_utc(r["close_time"]),
                        _to_float(r.get("open")),
                        _to_float(r.get("high")),
                        _to_float(r.get("low")),
                        _to_float(r.get("close")),
                        _to_float(r.get("volume")),
                        _to_float(r.get("quote_volume")),
                        int(r.get("trades") or 0),
                    )
                )

            sql = """
                INSERT INTO trading.market_klines(
                    symbol, interval, open_time, close_time,
                    open, high, low, close,
                    volume, quote_volume, trades
                ) VALUES %s
                ON CONFLICT (symbol, interval, open_time) DO NOTHING
            """

            execute_values(cur, sql, values, page_size=1000)
            # execute_values does not provide a reliable inserted rowcount when ON CONFLICT DO NOTHING.
            # Return the number of attempted inserts.
            return len(values)

        attempted = self._db.run(_q, retries=2, swallow=False)
        return int(attempted or 0)

    def read_latest(self, *, symbol: str, interval: str, limit: int) -> Optional[list[dict]]:
        symbol = str(symbol).upper().strip()
        interval = str(interval).strip()
        limit_n = max(1, int(limit))

        def _q(cur):
            cur.execute(
                """
                SELECT symbol, interval, open_time, close_time,
                       open, high, low, close,
                       volume, quote_volume, trades
                FROM trading.market_klines
                WHERE symbol = %s AND interval = %s
                ORDER BY open_time DESC
                LIMIT %s
                """,
                (symbol, interval, limit_n),
            )
            return cur.fetchall()

        rows = self._db.run(_q, retries=2, swallow=True)
        if rows is None:
            return None
        # Return ascending order (oldest->newest)
        out = list(rows)
        out.reverse()
        return out
