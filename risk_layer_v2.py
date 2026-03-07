import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from decimal import Decimal
from typing import Any, Optional, Callable


logger = logging.getLogger(__name__)


def _compute_atr_simple(
    *,
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> Optional[float]:
    """Computes a simple ATR (SMA of True Range) over the last `period` bars.

    Uses TR = max(high-low, abs(high-prev_close), abs(low-prev_close)).
    Returns None when there isn't enough data.
    """

    try:
        period = int(period)
    except Exception:
        period = 14

    if period <= 0:
        period = 14

    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None

    trs: list[float] = []
    for i in range(1, n):
        h = float(highs[i])
        l = float(lows[i])
        pc = float(closes[i - 1])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        if tr >= 0:
            trs.append(float(tr))

    if len(trs) < period:
        return None

    window = trs[-period:]
    atr = sum(window) / float(len(window)) if window else None
    if atr is None:
        return None
    if atr <= 0:
        return None
    return float(atr)


def _fetch_atr_from_db(
    *,
    db,
    symbol: str,
    interval: str,
    period: int,
    limit: int,
) -> Optional[float]:
    if db is None:
        return None

    sym = str(symbol).upper().strip()
    interval = str(interval).strip()
    period = int(period)
    limit = int(limit)

    if limit < period + 2:
        limit = period + 2

    def _q(cur):
        cur.execute(
            """
            SELECT high, low, close
            FROM trading.market_klines
            WHERE symbol = %s AND interval = %s
            ORDER BY open_time DESC
            LIMIT %s
            """,
            (sym, interval, limit),
        )
        rows = cur.fetchall() or []
        return rows

    rows = db.run(_q, retries=2, swallow=True)
    if not rows:
        return None

    # rows are newest-first; reverse to chronological
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for r in reversed(rows):
        try:
            highs.append(float(r.get("high") if isinstance(r, dict) else r[0]))
            lows.append(float(r.get("low") if isinstance(r, dict) else r[1]))
            closes.append(float(r.get("close") if isinstance(r, dict) else r[2]))
        except Exception:
            continue

    return _compute_atr_simple(highs=highs, lows=lows, closes=closes, period=period)


def _recovered_sl_tp_defaults(*, buy_price: float) -> tuple[float, float]:
    # Keep defaults aligned with bot_trading_v3_1.py
    take_profit_pct = _env_float("TAKE_PROFIT_PCT", 2.0)
    stop_loss_pct = _env_float("STOP_LOSS_PCT", 1.0)
    bp = float(buy_price)
    tp = bp * (1.0 + float(take_profit_pct) / 100.0)
    sl = bp * (1.0 - float(stop_loss_pct) / 100.0)
    return float(sl), float(tp)


def _recovered_sl_tp_from_atr(*, buy_price: float, atr: float) -> tuple[float, float, float, float, float]:
    sl_mult = _env_float("RECOVERED_SL_ATR_MULT", 2.0)
    tp_mult = _env_float("RECOVERED_TP_ATR_MULT", 2.0)
    trailing_mult = _env_float("RECOVERED_TRAILING_SL_ATR_MULT", 2.0)

    bp = float(buy_price)
    a = float(atr)
    sl = bp - float(sl_mult) * a
    tp = bp + float(tp_mult) * a

    # Clamp to sane bounds
    if sl <= 0:
        sl, _tp = _recovered_sl_tp_defaults(buy_price=bp)
        tp = max(tp, _tp)
    if tp <= bp:
        _sl, tp2 = _recovered_sl_tp_defaults(buy_price=bp)
        sl = min(sl, _sl)
        tp = max(tp, tp2)

    return float(sl), float(tp), float(a), float(sl_mult), float(tp_mult), float(trailing_mult)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


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


def _utc_today(ts: Optional[datetime] = None) -> date:
    ts = ts or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).date()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class HealthSnapshot:
    status: str
    db_connected: bool
    binance_connected: bool
    equity_current: Optional[float]
    drawdown_current: Optional[float]


class RiskEventLogger:
    def __init__(self, db):
        self._db = db

    def insert(self, *, event_type: str, severity: str, message: str, meta: Optional[dict] = None) -> None:
        if self._db is None:
            return

        def _q(cur):
            cur.execute(
                """
                INSERT INTO trading.risk_events(event_type, severity, message, meta)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (event_type, severity, message, json.dumps(meta or {})),
            )
            return True

        _ = self._db.run(_q, retries=2, swallow=True)


class GlobalRiskController:
    """Global equity kill switch.

    Stops the whole bot if global equity drawdown from peak exceeds threshold.

    State is persisted in trading.global_risk_state.
    """

    def __init__(
        self,
        *,
        db,
        stop_event,
        send_telegram: Callable[[str], None],
        event_logger: RiskEventLogger,
        enabled: bool | None = None,
        max_drawdown_frac: float | None = None,
    ):
        self._db = db
        self._stop_event = stop_event
        self._send_telegram = send_telegram
        self._event_logger = event_logger

        self.enabled = _env_bool("GLOBAL_KILL_SWITCH_ENABLED", True) if enabled is None else bool(enabled)
        self.max_drawdown_frac = _env_float("GLOBAL_MAX_DRAWDOWN_FRAC", 0.10) if max_drawdown_frac is None else float(max_drawdown_frac)

        self._lock = threading.Lock()
        self.peak_equity: Optional[float] = None
        self.current_equity: Optional[float] = None
        self.max_drawdown_reached: bool = False
        self.drawdown_frac: Optional[float] = None

        self._load_from_db()

    def _load_from_db(self) -> None:
        if self._db is None:
            return

        def _q(cur):
            cur.execute(
                """
                SELECT peak_equity, current_equity, drawdown_frac
                FROM trading.global_risk_state
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            )
            return cur.fetchone()

        row = self._db.run(_q, retries=2, swallow=True)
        if not row:
            return

        with self._lock:
            self.peak_equity = _to_float(row.get("peak_equity"))
            self.current_equity = _to_float(row.get("current_equity"))
            self.drawdown_frac = _to_float(row.get("drawdown_frac"))
            if self.drawdown_frac is not None and self.drawdown_frac >= float(self.max_drawdown_frac or 0.0):
                self.max_drawdown_reached = True

    def _persist_state(self, *, peak: float, current: float, dd: float) -> None:
        if self._db is None:
            return

        def _q(cur):
            cur.execute(
                """
                INSERT INTO trading.global_risk_state(peak_equity, current_equity, drawdown_frac, updated_at)
                VALUES (%s, %s, %s, now())
                """,
                (float(peak), float(current), float(dd)),
            )
            return True

        _ = self._db.run(_q, retries=2, swallow=True)

    def get_drawdown_frac(self) -> Optional[float]:
        with self._lock:
            return self.drawdown_frac

    def on_equity_snapshot(self, *, equity_total: float, when: Optional[datetime] = None) -> None:
        if not self.enabled:
            return
        if equity_total is None:
            return

        when = when or _now_utc()

        with self._lock:
            cur = float(equity_total)
            prev_peak = float(self.peak_equity) if self.peak_equity is not None else None
            peak = cur if prev_peak is None else max(prev_peak, cur)
            dd = 0.0
            if peak > 0:
                dd = max(0.0, (peak - cur) / peak)

            self.peak_equity = peak
            self.current_equity = cur
            self.drawdown_frac = dd

        self._persist_state(peak=peak, current=cur, dd=dd)

        if dd >= float(self.max_drawdown_frac):
            with self._lock:
                already = self.max_drawdown_reached
                self.max_drawdown_reached = True

            if not already:
                msg = (
                    f"🛑 GLOBAL KILL SWITCH: drawdown={dd*100:.2f}% >= {self.max_drawdown_frac*100:.2f}% "
                    f"peak={peak:.2f} current={cur:.2f}"
                )
                logger.critical(msg)
                try:
                    self._send_telegram(msg)
                except Exception:
                    logger.exception("Failed to send Telegram for Global Kill Switch")

                self._event_logger.insert(
                    event_type="GLOBAL_EQUITY_KILL_SWITCH",
                    severity="CRITICAL",
                    message=msg,
                    meta={"peak_equity": peak, "current_equity": cur, "drawdown_frac": dd, "ts": when.isoformat()},
                )
                try:
                    self._stop_event.set()
                except Exception:
                    logger.exception("Failed to set STOP_EVENT")


class SystemHealthMonitor:
    """Global circuit breaker for repeated critical failures."""

    def __init__(
        self,
        *,
        stop_event,
        send_telegram: Callable[[str], None],
        event_logger: RiskEventLogger,
        max_critical_errors: Optional[int] = None,
    ):
        self._stop_event = stop_event
        self._send_telegram = send_telegram
        self._event_logger = event_logger

        self.max_critical_errors = _env_int("MAX_CRITICAL_ERRORS", 5) if max_critical_errors is None else int(max_critical_errors)

        self._lock = threading.Lock()
        self.critical_errors_consecutive: int = 0
        self.last_success_ts: Optional[float] = None
        self.last_error_ts: Optional[float] = None

    def record_success(self) -> None:
        with self._lock:
            self.critical_errors_consecutive = 0
            self.last_success_ts = time.time()

    def record_critical(self, *, reason: str, meta: Optional[dict] = None) -> None:
        with self._lock:
            self.critical_errors_consecutive += 1
            n = self.critical_errors_consecutive
            self.last_error_ts = time.time()

        if n >= int(self.max_critical_errors):
            msg = f"🛑 CIRCUIT BREAKER: critical_errors={n} >= {self.max_critical_errors}. reason={reason}"
            logger.critical(msg)
            try:
                self._send_telegram(msg)
            except Exception:
                logger.exception("Failed to send Telegram for circuit breaker")

            self._event_logger.insert(
                event_type="GLOBAL_CIRCUIT_BREAKER",
                severity="CRITICAL",
                message=msg,
                meta={"reason": reason, "count": n, **(meta or {})},
            )
            try:
                self._stop_event.set()
            except Exception:
                logger.exception("Failed to set STOP_EVENT")


def insert_reconciliation_event(db, *, symbol: str, db_amount: Optional[float], exchange_amount: Optional[float], action_taken: str) -> None:
    if db is None:
        return

    sym = str(symbol).upper().strip()

    def _q(cur):
        cur.execute(
            """
            INSERT INTO trading.reconciliation_events(symbol, db_amount, exchange_amount, action_taken)
            VALUES (%s, %s, %s, %s)
            """,
            (sym, db_amount, exchange_amount, action_taken),
        )
        return True

    _ = db.run(_q, retries=2, swallow=True)


def _safe_get_free_balance(client_trade, asset: str) -> float:
    """Returns free balance for asset as float. Raises on errors."""
    bal = client_trade.get_asset_balance(asset=asset)
    if not bal:
        return 0.0
    free = bal.get("free")
    try:
        return float(free or 0.0)
    except Exception:
        return 0.0


def reconcile_positions_with_exchange(
    *,
    symbol: str,
    db,
    open_pos_repo,
    client_trade,
    get_live_price: Callable[[str], Optional[float]],
    event_logger: RiskEventLogger,
) -> None:
    """Reconciles DB open_positions vs real exchange balance for the symbol.

    Cases:
    A) DB has position(s), exchange base asset balance = 0 -> delete DB positions and log CRITICAL
    B) Exchange has base asset, DB has no position -> create RECOVERED position with current price
    C) Amount mismatch -> adjust DB amounts to match exchange and log WARNING
    """

    sym = str(symbol).upper().strip()
    base_asset = sym.replace("USDT", "") if sym.endswith("USDT") else sym

    db_positions = open_pos_repo.list_by_symbol(sym) or []
    db_amount = float(sum(float(p.get("amount") or 0.0) for p in db_positions))

    exch_amount = None
    try:
        exch_amount = float(_safe_get_free_balance(client_trade, base_asset))
    except Exception as e:
        logger.warning("reconcile: failed to fetch exchange balance sym=%s asset=%s err=%s", sym, base_asset, e)
        event_logger.insert(
            event_type="RECONCILIATION",
            severity="WARNING",
            message=f"reconcile: failed to fetch exchange balance {sym}/{base_asset}",
            meta={"symbol": sym, "asset": base_asset, "error": str(e)},
        )
        return

    # Tolerance to avoid churn due to tiny dust
    tol = 1e-10

    if db_amount > tol and (exch_amount or 0.0) <= tol:
        # Case A
        ok = open_pos_repo.replace_positions(sym, [])
        action = "DB_DELETED_MISSING_ON_EXCHANGE" if ok else "DB_DELETE_FAILED"
        insert_reconciliation_event(db, symbol=sym, db_amount=db_amount, exchange_amount=exch_amount, action_taken=action)
        msg = f"🚨 RECONCILIATION CRITICAL {sym}: DB amount={db_amount} but exchange balance=0. Deleted DB position(s)."
        logger.critical(msg)
        event_logger.insert(
            event_type="RECONCILIATION",
            severity="CRITICAL",
            message=msg,
            meta={"symbol": sym, "db_amount": db_amount, "exchange_amount": exch_amount, "action": action},
        )
        return

    if db_amount <= tol and (exch_amount or 0.0) > tol:
        # Case B
        px = None
        try:
            px = get_live_price(sym)
        except Exception:
            px = None
        px = float(px) if px is not None else None
        if px is None or px <= 0:
            msg = f"reconcile: cannot recover position for {sym} because live price unavailable"
            logger.warning(msg)
            event_logger.insert(
                event_type="RECONCILIATION",
                severity="WARNING",
                message=msg,
                meta={"symbol": sym, "exchange_amount": exch_amount},
            )
            return

        opened_at = _now_utc()

        # Institutional safety: never create an unmanaged RECOVERED position.
        atr_interval = os.getenv("RECOVERED_ATR_INTERVAL", os.getenv("TIMEFRAME", "1h"))
        atr_period = _env_int("RECOVERED_ATR_PERIOD", 14)
        atr_limit = _env_int("RECOVERED_ATR_LIMIT", 100)
        atr_now = _fetch_atr_from_db(db=db, symbol=sym, interval=atr_interval, period=atr_period, limit=atr_limit)

        stop_loss = None
        take_profit = None
        atr_entry = None
        sl_atr_mult = None
        tp_atr_mult = None
        trailing_sl_atr_mult = None

        if atr_now is not None and atr_now > 0:
            stop_loss, take_profit, atr_entry, sl_atr_mult, tp_atr_mult, trailing_sl_atr_mult = _recovered_sl_tp_from_atr(
                buy_price=px,
                atr=float(atr_now),
            )
        else:
            stop_loss, take_profit = _recovered_sl_tp_defaults(buy_price=px)

        pid = open_pos_repo.insert(
            {
                "symbol": sym,
                "buy_price": px,
                "amount": float(exch_amount),
                "regime": "RECOVERED",
                "opened_at": opened_at,
                "take_profit": float(take_profit) if take_profit is not None else None,
                "stop_loss": float(stop_loss) if stop_loss is not None else None,
                "atr_entry": float(atr_entry) if atr_entry is not None else None,
                "sl_atr_mult": float(sl_atr_mult) if sl_atr_mult is not None else None,
                "tp_atr_mult": float(tp_atr_mult) if tp_atr_mult is not None else None,
                "trailing_sl_atr_mult": float(trailing_sl_atr_mult) if trailing_sl_atr_mult is not None else None,
                "trailing_active": False,
                "max_price": float(px),
            }
        )
        action = "DB_INSERTED_RECOVERED" if pid is not None else "DB_INSERT_RECOVERED_FAILED"
        insert_reconciliation_event(db, symbol=sym, db_amount=db_amount, exchange_amount=exch_amount, action_taken=action)
        msg = (
            f"⚠️ RECONCILIATION {sym}: exchange has base asset={exch_amount}, DB empty -> inserted RECOVERED position @ {px} "
            f"SL={stop_loss} TP={take_profit} ATR={atr_now}"
        )
        logger.warning(msg)
        event_logger.insert(
            event_type="RECONCILIATION",
            severity="WARNING",
            message=msg,
            meta={
                "symbol": sym,
                "db_amount": db_amount,
                "exchange_amount": exch_amount,
                "buy_price": px,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_entry": atr_entry,
                "id": pid,
            },
        )
        return

    if db_amount > tol and (exch_amount or 0.0) > tol:
        # Case C: mismatch
        if abs(db_amount - float(exch_amount)) / max(db_amount, float(exch_amount), 1e-9) > 1e-6:
            if not db_positions:
                return

            # Adjust proportionally to preserve multi-lot positions
            factor = float(exch_amount) / db_amount if db_amount > 0 else 1.0
            new_positions = []
            for p in db_positions:
                p2 = dict(p)
                amt = float(p2.get("amount") or 0.0)
                p2["amount"] = float(amt * factor)
                new_positions.append(p2)

            ok = open_pos_repo.replace_positions(sym, new_positions)
            action = "DB_AMOUNT_ADJUSTED" if ok else "DB_ADJUST_FAILED"
            insert_reconciliation_event(db, symbol=sym, db_amount=db_amount, exchange_amount=exch_amount, action_taken=action)
            msg = f"RECONCILIATION WARNING {sym}: DB amount={db_amount} exchange={exch_amount} -> adjusted DB (factor={factor:.8f})"
            logger.warning(msg)
            event_logger.insert(
                event_type="RECONCILIATION",
                severity="WARNING",
                message=msg,
                meta={"symbol": sym, "db_amount": db_amount, "exchange_amount": exch_amount, "action": action, "factor": factor},
            )


def reconcile_worker(
    *,
    symbols: list[str],
    interval_s: float,
    stop_event,
    reconcile_fn: Callable[[str], None],
) -> None:
    """Runs reconciliation periodically in a dedicated thread."""
    next_run = 0.0
    while not stop_event.is_set():
        now = time.time()
        if now >= next_run:
            for sym in symbols:
                if stop_event.is_set():
                    break
                try:
                    reconcile_fn(sym)
                except Exception:
                    logger.exception("reconcile_worker: error sym=%s", sym)
            next_run = now + float(interval_s)

        stop_event.wait(1.0)


def upsert_risk_metrics_daily(
    db,
    *,
    day: date,
    equity_open: Optional[float],
    equity_close: Optional[float],
    daily_return_pct: Optional[float],
    max_drawdown_intraday: Optional[float],
    realized_pnl: Optional[float],
    floating_pnl: Optional[float],
    total_exposure_frac: Optional[float],
) -> None:
    if db is None:
        return

    def _q(cur):
        cur.execute(
            """
            INSERT INTO trading.risk_metrics_daily(
                date, equity_open, equity_close, daily_return_pct,
                max_drawdown_intraday, realized_pnl, floating_pnl, total_exposure_frac
            ) VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s
            )
            ON CONFLICT (date) DO UPDATE SET
                equity_open = COALESCE(EXCLUDED.equity_open, trading.risk_metrics_daily.equity_open),
                equity_close = COALESCE(EXCLUDED.equity_close, trading.risk_metrics_daily.equity_close),
                daily_return_pct = COALESCE(EXCLUDED.daily_return_pct, trading.risk_metrics_daily.daily_return_pct),
                max_drawdown_intraday = GREATEST(
                    COALESCE(trading.risk_metrics_daily.max_drawdown_intraday, 0),
                    COALESCE(EXCLUDED.max_drawdown_intraday, 0)
                ),
                realized_pnl = COALESCE(EXCLUDED.realized_pnl, trading.risk_metrics_daily.realized_pnl),
                floating_pnl = COALESCE(EXCLUDED.floating_pnl, trading.risk_metrics_daily.floating_pnl),
                total_exposure_frac = COALESCE(EXCLUDED.total_exposure_frac, trading.risk_metrics_daily.total_exposure_frac),
                created_at = now()
            """,
            (
                day,
                equity_open,
                equity_close,
                daily_return_pct,
                max_drawdown_intraday,
                realized_pnl,
                floating_pnl,
                total_exposure_frac,
            ),
        )
        return True

    _ = db.run(_q, retries=2, swallow=True)


def _fetch_equity_open_close(db, day: date) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (open, close, peak) equity for the given day from equity_snapshots."""
    if db is None:
        return None, None, None

    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start.replace(hour=23, minute=59, second=59)

    def _q(cur):
        cur.execute(
            """
            SELECT
                (SELECT equity_total FROM trading.equity_snapshots WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC LIMIT 1) AS equity_open,
                (SELECT equity_total FROM trading.equity_snapshots WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp DESC LIMIT 1) AS equity_close,
                (SELECT MAX(equity_total) FROM trading.equity_snapshots WHERE timestamp >= %s AND timestamp <= %s) AS equity_peak
            """,
            (start, end, start, end, start, end),
        )
        return cur.fetchone()

    row = db.run(_q, retries=2, swallow=True)
    if not row:
        return None, None, None

    return _to_float(row.get("equity_open")), _to_float(row.get("equity_close")), _to_float(row.get("equity_peak"))


def _fetch_realized_pnl(db, day: date) -> Optional[float]:
    if db is None:
        return None

    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start.replace(hour=23, minute=59, second=59)

    def _q(cur):
        cur.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS pnl
            FROM trading.trade_history
            WHERE executed_at >= %s AND executed_at <= %s
            """,
            (start, end),
        )
        return cur.fetchone()

    row = db.run(_q, retries=2, swallow=True)
    if not row:
        return None
    return float(_to_float(row.get("pnl")) or 0.0)


def risk_metrics_worker(
    *,
    symbols: list[str],
    stop_event,
    interval_s: float,
    compute_equity_total: Callable[[], float],
    get_open_positions: Callable[[], list[dict]],
    get_live_price: Callable[[str], Optional[float]],
    db,
) -> None:
    """Institutional risk metrics snapshot worker (every 5 minutes)."""
    last_day = _utc_today()

    while not stop_event.is_set():
        now = _now_utc()
        today = _utc_today(now)

        try:
            equity = float(compute_equity_total())
        except Exception:
            logger.exception("risk_metrics_worker: compute_equity_total failed")
            equity = None

        floating_pnl = None
        exposure_frac = None

        try:
            positions = get_open_positions() or []
            total_value = 0.0
            total_pnl = 0.0
            for p in positions:
                sym = str(p.get("symbol") or "").upper().strip()
                amt = float(p.get("amount") or 0.0)
                buy = float(p.get("buy_price") or 0.0)
                if not sym or amt <= 0:
                    continue
                px = None
                try:
                    px = get_live_price(sym)
                except Exception:
                    px = None
                if px is None or px <= 0:
                    continue
                total_value += float(px) * amt
                total_pnl += (float(px) - buy) * amt

            floating_pnl = float(total_pnl)
            if equity and equity > 0:
                exposure_frac = float(total_value / equity)
        except Exception:
            logger.exception("risk_metrics_worker: compute floating/exposure failed")

        equity_open, equity_close, equity_peak = _fetch_equity_open_close(db, today)
        realized_pnl = _fetch_realized_pnl(db, today)

        dd_intraday = None
        if equity_peak and equity and equity_peak > 0:
            dd_intraday = max(0.0, (float(equity_peak) - float(equity)) / float(equity_peak))

        daily_return_pct = None
        if equity_open and equity_close and float(equity_open) != 0:
            daily_return_pct = (float(equity_close) - float(equity_open)) / float(equity_open)

        upsert_risk_metrics_daily(
            db,
            day=today,
            equity_open=equity_open,
            equity_close=equity_close,
            daily_return_pct=daily_return_pct,
            max_drawdown_intraday=dd_intraday,
            realized_pnl=realized_pnl,
            floating_pnl=floating_pnl,
            total_exposure_frac=exposure_frac,
        )

        # Day rollover consolidation is naturally handled by the per-day upsert.
        last_day = today

        stop_event.wait(float(interval_s))


def start_health_server(
    *,
    host: str,
    port: int,
    stop_event,
    db,
    health_monitor: SystemHealthMonitor,
    global_risk: Optional[GlobalRiskController],
    compute_equity_total: Callable[[], float],
    ping_binance: Callable[[], bool],
) -> None:
    """Starts a minimal Flask /health endpoint in a daemon thread."""

    try:
        from flask import Flask, jsonify
    except Exception as e:
        logger.error("Health server not started (Flask missing): %s", e)
        return

    app = Flask(__name__)

    def _db_ok() -> bool:
        if db is None:
            return False

        def _q(cur):
            cur.execute("SELECT 1")
            return True

        ok = db.run(_q, retries=1, swallow=True)
        return bool(ok)

    @app.get("/health")
    def health():
        stopped = bool(stop_event.is_set())

        db_connected = _db_ok()
        try:
            binance_connected = bool(ping_binance())
        except Exception:
            binance_connected = False

        equity_current = None
        try:
            equity_current = float(compute_equity_total())
        except Exception:
            equity_current = None

        drawdown_current = None
        if global_risk is not None:
            drawdown_current = global_risk.get_drawdown_frac()

        if stopped:
            status = "STOPPED"
        elif not db_connected or not binance_connected:
            status = "DEGRADED"
        else:
            status = "OK"

        return jsonify(
            {
                "status": status,
                "db_connected": db_connected,
                "binance_connected": binance_connected,
                "equity_current": equity_current,
                "drawdown_current": drawdown_current,
            }
        )

    # Run Flask server; keep it simple and avoid the reloader in a threaded context.
    app.run(host=host, port=int(port), threaded=True, debug=False, use_reloader=False)
