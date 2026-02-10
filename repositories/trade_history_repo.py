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


class TradeHistoryRepository:
    def __init__(self, db: PostgresDatabase):
        self._db = db

    def insert_trade(
        self,
        *,
        symbol: str,
        side: str,
        reason: Optional[str],
        buy_price: Optional[float],
        sell_price: Optional[float],
        amount: Optional[float],
        realized_pnl: Optional[float],
        realized_pnl_pct: Optional[float],
        volatility: Optional[float],
        rsi: Optional[float],
        stochrsi_k: Optional[float],
        stochrsi_d: Optional[float],
        regime: Optional[str],
        executed_at: Any,
    ) -> Optional[int]:
        symbol = str(symbol).upper().strip()
        side = str(side).upper().strip()
        if side not in ("BUY", "SELL"):
            logger.warning("trade_history.insert_trade: invalid side=%s, forcing SELL/BUY", side)

        executed_at_dt = _to_dt(executed_at)

        def _q(cur):
            cur.execute(
                """
                INSERT INTO trading.trade_history(
                    symbol, side, reason,
                    buy_price, sell_price, amount,
                    realized_pnl, realized_pnl_pct,
                    volatility, rsi, stochrsi_k, stochrsi_d,
                    regime, executed_at
                ) VALUES (
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,%s,%s,
                    %s,%s
                )
                RETURNING id
                """,
                (
                    symbol,
                    side,
                    reason,
                    _to_float(buy_price),
                    _to_float(sell_price),
                    _to_float(amount),
                    _to_float(realized_pnl),
                    _to_float(realized_pnl_pct),
                    _to_float(volatility),
                    _to_float(rsi),
                    _to_float(stochrsi_k),
                    _to_float(stochrsi_d),
                    regime,
                    executed_at_dt,
                ),
            )
            row = cur.fetchone()
            return int(row["id"]) if row and row.get("id") is not None else None

        return self._db.run(_q, retries=2, swallow=True)
