import logging
from dataclasses import dataclass
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


def _to_bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _to_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    # pandas Timestamp supports to_pydatetime
    try:
        dt = v.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


@dataclass
class OpenPosition:
    id: int
    symbol: str
    buy_price: float
    amount: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    regime: Optional[str]
    atr_entry: Optional[float]
    tp_atr_mult: Optional[float]
    sl_atr_mult: Optional[float]
    trailing_sl_atr_mult: Optional[float]
    trailing_active: bool
    max_price: Optional[float]
    opened_at: datetime
    updated_at: datetime


class OpenPositionsRepository:
    def __init__(self, db: PostgresDatabase):
        self._db = db

    def list_by_symbol(self, symbol: str) -> Optional[list[dict]]:
        symbol = str(symbol).upper().strip()

        def _q(cur):
            cur.execute(
                """
                SELECT id, symbol, buy_price, amount, take_profit, stop_loss, regime,
                       atr_entry, tp_atr_mult, sl_atr_mult, trailing_sl_atr_mult,
                       trailing_active, max_price, opened_at, updated_at
                FROM trading.open_positions
                WHERE symbol = %s
                ORDER BY opened_at ASC, id ASC
                """,
                (symbol,),
            )
            return cur.fetchall()

        rows = self._db.run(_q, retries=2, swallow=True)
        if rows is None:
            return None
        return [self._row_to_dict(r) for r in rows]

    def list_all(self) -> Optional[list[dict]]:
        def _q(cur):
            cur.execute(
                """
                SELECT id, symbol, buy_price, amount, take_profit, stop_loss, regime,
                       atr_entry, tp_atr_mult, sl_atr_mult, trailing_sl_atr_mult,
                       trailing_active, max_price, opened_at, updated_at
                FROM trading.open_positions
                ORDER BY opened_at ASC, id ASC
                """
            )
            return cur.fetchall()

        rows = self._db.run(_q, retries=2, swallow=True)
        if rows is None:
            return None
        return [self._row_to_dict(r) for r in rows]

    def replace_positions(self, symbol: str, positions: list[dict]) -> bool:
        """Reconciles DB rows for `symbol` to match the provided list.

        - Positions with `id` are updated.
        - Positions without `id` are inserted.
        - Any existing DB position id not present in the list is deleted.

        This keeps legacy code paths working (load -> mutate -> save) while using PostgreSQL.
        """
        symbol = str(symbol).upper().strip()

        def _q(cur):
            cur.execute("SELECT id FROM trading.open_positions WHERE symbol = %s", (symbol,))
            existing = {int(r["id"]) for r in (cur.fetchall() or []) if r.get("id") is not None}

            keep: set[int] = set()
            for p in positions or []:
                pid = p.get("id")
                if pid is not None:
                    try:
                        pid_int = int(pid)
                    except Exception:
                        pid_int = None
                    if pid_int is not None:
                        keep.add(pid_int)
                        # Update only known mutable fields
                        fields = dict(p)
                        fields.pop("id", None)
                        fields.pop("symbol", None)
                        self._update_fields_in_cursor(cur, pid_int, fields)
                        continue

                # Insert new
                to_ins = dict(p)
                to_ins["symbol"] = symbol
                new_id = self._insert_in_cursor(cur, to_ins)
                if new_id is not None:
                    keep.add(int(new_id))

            to_delete = list(existing - keep)
            if to_delete:
                cur.execute(
                    "DELETE FROM trading.open_positions WHERE symbol = %s AND id = ANY(%s::bigint[])",
                    (symbol, to_delete),
                )
            return True

        return bool(self._db.run(_q, retries=2, swallow=True) or False)

    def insert(self, position: dict) -> Optional[int]:
        """Inserts a new open position. Returns id or None on error."""
        symbol = str(position.get("symbol") or "").upper().strip()
        if not symbol:
            symbol = str(position.get("_symbol") or "").upper().strip()
        if not symbol:
            logger.error("open_positions.insert: missing symbol")
            return None

        buy_price = _to_float(position.get("buy_price"))
        amount = _to_float(position.get("amount"))
        if buy_price is None or amount is None:
            logger.error("open_positions.insert: missing buy_price/amount")
            return None

        opened_at = _to_dt(position.get("opened_at") or position.get("timestamp"))

        def _q(cur):
            cur.execute(
                """
                INSERT INTO trading.open_positions(
                    symbol, buy_price, amount, take_profit, stop_loss, regime,
                    atr_entry, tp_atr_mult, sl_atr_mult, trailing_sl_atr_mult,
                    trailing_active, max_price, opened_at, updated_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s, now()
                )
                RETURNING id
                """,
                (
                    symbol,
                    buy_price,
                    amount,
                    _to_float(position.get("take_profit")),
                    _to_float(position.get("stop_loss")),
                    position.get("regime"),
                    _to_float(position.get("atr_entry")),
                    _to_float(position.get("tp_atr_mult")),
                    _to_float(position.get("sl_atr_mult")),
                    _to_float(position.get("trailing_sl_atr_mult")),
                    _to_bool(position.get("trailing_active")),
                    _to_float(position.get("max_price")),
                    opened_at,
                ),
            )
            row = cur.fetchone()
            return int(row["id"]) if row and row.get("id") is not None else None

        return self._db.run(_q, retries=2, swallow=True)

    def update_fields(self, position_id: int, fields: dict) -> bool:
        allowed = {
            "take_profit",
            "stop_loss",
            "regime",
            "atr_entry",
            "tp_atr_mult",
            "sl_atr_mult",
            "trailing_sl_atr_mult",
            "trailing_active",
            "max_price",
            "buy_price",
            "amount",
            "opened_at",
        }
        sets = []
        params: list[Any] = []

        for k, v in (fields or {}).items():
            if k not in allowed:
                continue
            sets.append(f"{k} = %s")
            if k in ("take_profit", "stop_loss", "atr_entry", "tp_atr_mult", "sl_atr_mult", "trailing_sl_atr_mult", "max_price", "buy_price", "amount"):
                params.append(_to_float(v))
            elif k == "trailing_active":
                params.append(_to_bool(v))
            elif k == "opened_at":
                params.append(_to_dt(v))
            else:
                params.append(v)

        if not sets:
            return True

        params.append(int(position_id))

        sql = "UPDATE trading.open_positions SET " + ", ".join(sets) + ", updated_at = now() WHERE id = %s"

        def _q(cur):
            cur.execute(sql, tuple(params))
            return cur.rowcount > 0

        return bool(self._db.run(_q, retries=2, swallow=True) or False)

    def _update_fields_in_cursor(self, cur, position_id: int, fields: dict) -> None:
        """Same as update_fields but reuses an existing cursor/transaction."""
        allowed = {
            "take_profit",
            "stop_loss",
            "regime",
            "atr_entry",
            "tp_atr_mult",
            "sl_atr_mult",
            "trailing_sl_atr_mult",
            "trailing_active",
            "max_price",
            "buy_price",
            "amount",
            "opened_at",
        }
        sets = []
        params: list[Any] = []

        for k, v in (fields or {}).items():
            if k not in allowed:
                continue
            sets.append(f"{k} = %s")
            if k in ("take_profit", "stop_loss", "atr_entry", "tp_atr_mult", "sl_atr_mult", "trailing_sl_atr_mult", "max_price", "buy_price", "amount"):
                params.append(_to_float(v))
            elif k == "trailing_active":
                params.append(_to_bool(v))
            elif k == "opened_at":
                params.append(_to_dt(v))
            else:
                params.append(v)

        if not sets:
            return

        params.append(int(position_id))
        sql = "UPDATE trading.open_positions SET " + ", ".join(sets) + ", updated_at = now() WHERE id = %s"
        cur.execute(sql, tuple(params))

    def _insert_in_cursor(self, cur, position: dict) -> Optional[int]:
        symbol = str(position.get("symbol") or "").upper().strip()
        if not symbol:
            return None
        buy_price = _to_float(position.get("buy_price"))
        amount = _to_float(position.get("amount"))
        if buy_price is None or amount is None:
            return None
        opened_at = _to_dt(position.get("opened_at") or position.get("timestamp"))

        cur.execute(
            """
            INSERT INTO trading.open_positions(
                symbol, buy_price, amount, take_profit, stop_loss, regime,
                atr_entry, tp_atr_mult, sl_atr_mult, trailing_sl_atr_mult,
                trailing_active, max_price, opened_at, updated_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s, now()
            )
            RETURNING id
            """,
            (
                symbol,
                buy_price,
                amount,
                _to_float(position.get("take_profit")),
                _to_float(position.get("stop_loss")),
                position.get("regime"),
                _to_float(position.get("atr_entry")),
                _to_float(position.get("tp_atr_mult")),
                _to_float(position.get("sl_atr_mult")),
                _to_float(position.get("trailing_sl_atr_mult")),
                _to_bool(position.get("trailing_active")),
                _to_float(position.get("max_price")),
                opened_at,
            ),
        )
        row = cur.fetchone()
        return int(row["id"]) if row and row.get("id") is not None else None

    def delete(self, position_id: int) -> bool:
        def _q(cur):
            cur.execute("DELETE FROM trading.open_positions WHERE id = %s", (int(position_id),))
            return cur.rowcount > 0

        return bool(self._db.run(_q, retries=2, swallow=True) or False)

    def delete_by_symbol(self, symbol: str) -> int:
        symbol = str(symbol).upper().strip()

        def _q(cur):
            cur.execute("DELETE FROM trading.open_positions WHERE symbol = %s", (symbol,))
            return int(cur.rowcount or 0)

        return int(self._db.run(_q, retries=2, swallow=True) or 0)

    def _row_to_dict(self, row: dict) -> dict:
        opened_at = row.get("opened_at")
        if isinstance(opened_at, datetime):
            ts_str = opened_at.isoformat()
        else:
            ts_str = str(opened_at)

        return {
            "id": int(row.get("id")),
            "symbol": row.get("symbol"),
            "buy_price": _to_float(row.get("buy_price")) or 0.0,
            "amount": _to_float(row.get("amount")) or 0.0,
            "take_profit": _to_float(row.get("take_profit")),
            "stop_loss": _to_float(row.get("stop_loss")),
            "regime": row.get("regime"),
            "atr_entry": _to_float(row.get("atr_entry")),
            "tp_atr_mult": _to_float(row.get("tp_atr_mult")),
            "sl_atr_mult": _to_float(row.get("sl_atr_mult")),
            "trailing_sl_atr_mult": _to_float(row.get("trailing_sl_atr_mult")),
            "trailing_active": bool(row.get("trailing_active")),
            "max_price": _to_float(row.get("max_price")),
            "opened_at": row.get("opened_at"),
            "updated_at": row.get("updated_at"),
            # Compatibility with legacy bot fields
            "timestamp": ts_str,
        }
