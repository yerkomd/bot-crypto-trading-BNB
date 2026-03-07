import logging
import json
from datetime import datetime, timezone
from datetime import date as _date
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


def _to_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    try:
        dt = v.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


class EquitySnapshotsRepository:
    def __init__(self, db: PostgresDatabase):
        self._db = db

    def insert_snapshot(
        self,
        *,
        timestamp: Any,
        equity_total: float,
        usdt_balance: float,
        positions_value: float,
        positions_json: dict,
    ) -> bool:
        ts = _to_dt(timestamp)

        def _json_default(o: Any):
            # Make nested structures JSON-serializable (psycopg2 Json uses json.dumps under the hood).
            if isinstance(o, datetime):
                dt = o
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            if isinstance(o, _date):
                return o.isoformat()
            if isinstance(o, Decimal):
                return float(o)
            # pandas.Timestamp and similar
            try:
                to_pydt = getattr(o, "to_pydatetime", None)
                if callable(to_pydt):
                    dt = to_pydt()
                    if isinstance(dt, datetime):
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.isoformat()
            except Exception:
                pass
            return str(o)

        def _q(cur):
            # psycopg2 Json adapter
            from psycopg2.extras import Json

            cur.execute(
                """
                INSERT INTO trading.equity_snapshots(
                    timestamp, equity_total, usdt_balance, positions_value, positions_json
                ) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (timestamp) DO NOTHING
                """,
                (
                    ts,
                    _to_float(equity_total),
                    _to_float(usdt_balance),
                    _to_float(positions_value),
                    Json(positions_json, dumps=lambda obj: json.dumps(obj, default=_json_default, ensure_ascii=False)),
                ),
            )
            return True

        return bool(self._db.run(_q, retries=2, swallow=True) or False)
