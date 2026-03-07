import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from services.market_klines_service import read_klines_df


logger = logging.getLogger(__name__)


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


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_log_block(reason: str) -> None:
    logger.warning("RISK_V3_BLOCK: %s", reason)


@dataclass(frozen=True)
class _TimedValue:
    ts: float
    value: object


class VolatilityPositionSizer:
    def __init__(self) -> None:
        self.enabled = _env_bool("RISK_V3_POSITION_SIZER_ENABLED", False)
        self.risk_per_trade_frac = _env_float("RISK_PER_TRADE_FRAC", 0.01)
        self.max_position_cap_frac = _env_float("MAX_POSITION_CAP_FRAC", 0.10)
        self.min_position_usdt = _env_float("MIN_POSITION_USDT", 10.0)
        self.atr_multiplier = _env_float("POSITION_SIZER_ATR_MULTIPLIER", 1.0)
        self.atr_window = _env_int("POSITION_SIZER_ATR_WINDOW", 14)

    def compute_position_size(self, symbol, equity_total, atr, price) -> float:
        if not self.enabled:
            return 0.0
        try:
            eq = float(equity_total)
            atr_v = float(atr)
            px = float(price)
        except Exception:
            return 0.0

        if eq <= 0 or atr_v <= 0 or px <= 0:
            return 0.0

        atr_mult = float(self.atr_multiplier) if float(self.atr_multiplier) > 0 else 1.0
        risk_frac = max(0.0, float(self.risk_per_trade_frac))

        position_size_usdt = (eq * risk_frac) / (atr_v * atr_mult)
        cap_usdt = eq * max(0.0, float(self.max_position_cap_frac))
        if cap_usdt > 0:
            position_size_usdt = min(position_size_usdt, cap_usdt)

        if position_size_usdt < float(self.min_position_usdt):
            return 0.0

        qty_base = position_size_usdt / px
        return float(max(0.0, qty_base))


class PortfolioCorrelationRisk:
    def __init__(self, *, db, symbols: list[str]) -> None:
        self._db = db
        self._symbols = sorted({str(s).upper().strip() for s in (symbols or []) if str(s).strip()})
        self._lock = threading.Lock()

        self.enabled = _env_bool("RISK_V3_CORRELATION_ENABLED", False)
        self.window_days = _env_int("CORRELATION_WINDOW_DAYS", 30)
        self.corr_threshold = _env_float("CORRELATION_THRESHOLD", 0.8)
        self.max_combined_exposure_frac = _env_float("CORRELATION_MAX_COMBINED_EXPOSURE_FRAC", 0.25)
        self.interval = os.getenv("CORRELATION_INTERVAL", "1d")
        self.refresh_s = _env_float("CORRELATION_CACHE_SECONDS", 300.0)

        self._cache: Optional[_TimedValue] = None

    def _compute_corr_matrix(self, symbols: list[str]) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()

        ret_by_sym: dict[str, pd.Series] = {}
        bars = max(60, int(self.window_days) + 10)
        for sym in symbols:
            df = read_klines_df(db=self._db, symbol=sym, interval=self.interval, limit=bars)
            if df is None or df.empty or "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if close.shape[0] < int(self.window_days) + 1:
                continue
            ratio = close / close.shift(1)
            ratio = pd.to_numeric(ratio, errors="coerce")
            ratio = ratio.where(ratio > 0)
            log_ret = ratio.dropna().apply(lambda x: float(math.log(float(x))))
            if log_ret.shape[0] < int(self.window_days):
                continue
            ret_by_sym[sym] = log_ret.tail(int(self.window_days)).reset_index(drop=True)

        if len(ret_by_sym) < 2:
            return pd.DataFrame()

        data = pd.DataFrame(ret_by_sym).dropna(how="any")
        if data.empty:
            return pd.DataFrame()
        return data.corr()

    def _get_corr_matrix(self, symbols: list[str]) -> pd.DataFrame:
        now = time.time()
        with self._lock:
            cache = self._cache
            if cache is not None and (now - cache.ts) <= float(self.refresh_s):
                return cache.value if isinstance(cache.value, pd.DataFrame) else pd.DataFrame()

        mat = self._compute_corr_matrix(symbols)

        with self._lock:
            self._cache = _TimedValue(ts=now, value=mat)
        return mat

    def _position_notional(self, p: dict) -> float:
        try:
            amt = float(p.get("amount") or 0.0)
            if amt <= 0:
                return 0.0
            px = _to_float(p.get("current_price"))
            if px is None or px <= 0:
                px = _to_float(p.get("buy_price"))
            if px is None or px <= 0:
                return 0.0
            return float(amt * px)
        except Exception:
            return 0.0

    def can_open(self, symbol, current_positions, equity_total) -> tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None

        try:
            eq = float(equity_total)
        except Exception:
            return True, None
        if eq <= 0:
            return True, None

        try:
            sym = str(symbol).upper().strip()
            positions = list(current_positions or [])
            symbols_set = sorted({sym, *self._symbols, *[str(p.get("symbol") or "").upper().strip() for p in positions]})
            corr = self._get_corr_matrix(symbols_set)
            if corr.empty or sym not in corr.index:
                return True, None

            correlated: set[str] = {sym}
            for other in corr.columns:
                if other == sym:
                    continue
                try:
                    c = float(corr.loc[sym, other])
                except Exception:
                    continue
                if c >= float(self.corr_threshold):
                    correlated.add(str(other).upper())

            if len(correlated) <= 1:
                return True, None

            combined = 0.0
            for p in positions:
                psym = str(p.get("symbol") or "").upper().strip()
                if psym in correlated:
                    combined += self._position_notional(p)

            frac = combined / eq if eq > 0 else 0.0
            if frac > float(self.max_combined_exposure_frac):
                reason = (
                    f"correlation_exposure={frac:.4f} > max={self.max_combined_exposure_frac:.4f} "
                    f"for cluster={sorted(correlated)}"
                )
                _safe_log_block(reason)
                return False, reason
            return True, None
        except Exception as e:
            logger.warning("PortfolioCorrelationRisk failed open: %s", e)
            return True, None


class IntradayVaRMonitor:
    def __init__(self, *, db) -> None:
        self._db = db
        self._lock = threading.Lock()

        self.enabled = _env_bool("RISK_V3_VAR_ENABLED", False)
        self.window_days = _env_int("VAR_WINDOW_DAYS", 30)
        self.confidence = _env_float("VAR_CONFIDENCE", 0.95)
        self.max_var_frac = _env_float("MAX_VAR_FRAC", 0.05)
        self.refresh_s = _env_float("VAR_CACHE_SECONDS", 300.0)

        self._cache: Optional[_TimedValue] = None

    def _compute_var_frac(self) -> Optional[float]:
        if self._db is None:
            return None

        lookback_days = max(int(self.window_days) + 10, 60)

        def _q(cur):
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        date_trunc('day', timestamp) AS day,
                        equity_total,
                        ROW_NUMBER() OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp DESC) AS rn
                    FROM trading.equity_snapshots
                    WHERE timestamp >= now() - (%s * interval '1 day')
                )
                SELECT day, equity_total
                FROM ranked
                WHERE rn = 1
                ORDER BY day ASC
                """,
                (int(lookback_days),),
            )
            return cur.fetchall() or []

        rows = self._db.run(_q, retries=2, swallow=True)
        if not rows:
            return None

        closes = []
        for r in rows:
            v = _to_float(r.get("equity_total") if isinstance(r, dict) else r[1])
            if v is not None and v > 0:
                closes.append(float(v))
        if len(closes) < int(self.window_days) + 1:
            return None

        s = pd.Series(closes, dtype="float64")
        rets = s.pct_change().dropna().tail(int(self.window_days))
        if rets.empty:
            return None

        alpha = max(0.0, min(1.0, 1.0 - float(self.confidence)))
        var = float(abs(rets.quantile(alpha)))
        return var

    def _get_var_frac(self) -> Optional[float]:
        now = time.time()
        with self._lock:
            cache = self._cache
            if cache is not None and (now - cache.ts) <= float(self.refresh_s):
                return _to_float(cache.value)

        var_frac = self._compute_var_frac()
        with self._lock:
            self._cache = _TimedValue(ts=now, value=var_frac)
        return var_frac

    def can_open(self, equity_total) -> tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None

        try:
            eq = float(equity_total)
        except Exception:
            eq = 0.0
        if eq <= 0:
            return True, None

        try:
            var_frac = self._get_var_frac()
            if var_frac is None:
                return True, None
            if float(var_frac) > float(self.max_var_frac):
                reason = f"intraday_var={var_frac:.4f} > max_var_frac={self.max_var_frac:.4f}"
                _safe_log_block(reason)
                return False, reason
            return True, None
        except Exception as e:
            logger.warning("IntradayVaRMonitor failed open: %s", e)
            return True, None


class SlippageMonitor:
    def __init__(self, *, stop_event, event_logger=None, send_telegram: Optional[Callable[[str], None]] = None) -> None:
        self._stop_event = stop_event
        self._event_logger = event_logger
        self._send_telegram = send_telegram
        self._lock = threading.Lock()

        self.enabled = _env_bool("RISK_V3_SLIPPAGE_ENABLED", False)
        self.max_slippage_frac = _env_float("SLIPPAGE_MAX_FRAC", 0.005)
        self.max_consecutive = _env_int("SLIPPAGE_MAX_CONSECUTIVE", 3)
        self.consecutive_breaches = 0

    def record_fill(self, expected_price, fill_price) -> None:
        if not self.enabled:
            return

        try:
            exp = float(expected_price)
            fill = float(fill_price)
        except Exception:
            return
        if exp <= 0 or fill <= 0:
            return

        slippage = abs(fill - exp) / exp
        trigger_stop = False

        with self._lock:
            if slippage > float(self.max_slippage_frac):
                self.consecutive_breaches += 1
                n = int(self.consecutive_breaches)
                logger.warning(
                    "Slippage breach: slippage=%.4f%% > %.4f%% (consecutive=%d)",
                    slippage * 100.0,
                    float(self.max_slippage_frac) * 100.0,
                    n,
                )
                if n >= int(self.max_consecutive):
                    trigger_stop = True
            else:
                self.consecutive_breaches = 0

        if slippage > float(self.max_slippage_frac):
            msg = (
                f"RISK_V3_BLOCK: slippage={slippage:.4%} > max={float(self.max_slippage_frac):.4%} "
                f"consecutive={self.consecutive_breaches}"
            )
            try:
                if self._event_logger is not None:
                    self._event_logger.insert(
                        event_type="SLIPPAGE",
                        severity="WARNING",
                        message=msg,
                        meta={
                            "expected_price": exp,
                            "fill_price": fill,
                            "slippage_frac": slippage,
                            "consecutive": int(self.consecutive_breaches),
                        },
                    )
            except Exception:
                pass

        if trigger_stop:
            msg = (
                f"RISK_V3_BLOCK: slippage_breaches={self.consecutive_breaches} "
                f">= max_consecutive={self.max_consecutive}. Activating STOP_EVENT"
            )
            logger.critical(msg)
            try:
                if self._event_logger is not None:
                    self._event_logger.insert(
                        event_type="SLIPPAGE_KILL_SWITCH",
                        severity="CRITICAL",
                        message=msg,
                        meta={"consecutive": int(self.consecutive_breaches)},
                    )
            except Exception:
                pass

            try:
                if callable(self._send_telegram):
                    self._send_telegram(msg)
            except Exception:
                pass

            try:
                self._stop_event.set()
            except Exception:
                pass


class EquityRegimeFilter:
    def __init__(self, *, db) -> None:
        self._db = db
        self._lock = threading.Lock()

        self.enabled = _env_bool("RISK_V3_EQUITY_REGIME_ENABLED", False)
        self.mode = str(os.getenv("EQUITY_REGIME_MODE", "reduce")).strip().lower()
        self.reduction_frac = _env_float("EQUITY_REGIME_REDUCTION_FRAC", 0.5)
        self.ema_fast = _env_int("EQUITY_EMA_FAST", 50)
        self.ema_slow = _env_int("EQUITY_EMA_SLOW", 200)
        self.refresh_s = _env_float("EQUITY_REGIME_CACHE_SECONDS", 300.0)

        self._cache: Optional[_TimedValue] = None

    def _compute_regime_bearish(self) -> Optional[bool]:
        if self._db is None:
            return None

        lookback_days = max(int(self.ema_slow) + 30, 240)

        def _q(cur):
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        date_trunc('day', timestamp) AS day,
                        equity_total,
                        ROW_NUMBER() OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp DESC) AS rn
                    FROM trading.equity_snapshots
                    WHERE timestamp >= now() - (%s * interval '1 day')
                )
                SELECT day, equity_total
                FROM ranked
                WHERE rn = 1
                ORDER BY day ASC
                """,
                (int(lookback_days),),
            )
            return cur.fetchall() or []

        rows = self._db.run(_q, retries=2, swallow=True)
        if not rows:
            return None

        closes = []
        for r in rows:
            v = _to_float(r.get("equity_total") if isinstance(r, dict) else r[1])
            if v is not None and v > 0:
                closes.append(float(v))
        if len(closes) < int(self.ema_slow):
            return None

        s = pd.Series(closes, dtype="float64")
        ema_fast = s.ewm(span=max(2, int(self.ema_fast)), adjust=False).mean().iloc[-1]
        ema_slow = s.ewm(span=max(3, int(self.ema_slow)), adjust=False).mean().iloc[-1]
        return bool(float(ema_fast) < float(ema_slow))

    def _is_bearish(self) -> Optional[bool]:
        now = time.time()
        with self._lock:
            cache = self._cache
            if cache is not None and (now - cache.ts) <= float(self.refresh_s):
                if isinstance(cache.value, bool) or cache.value is None:
                    return cache.value

        bearish = self._compute_regime_bearish()
        with self._lock:
            self._cache = _TimedValue(ts=now, value=bearish)
        return bearish

    def can_open(self) -> tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None
        if self.mode != "block":
            return True, None

        try:
            bearish = self._is_bearish()
            if bearish is True:
                reason = "equity_regime_bearish (EMA50 < EMA200) in block mode"
                _safe_log_block(reason)
                return False, reason
            return True, None
        except Exception as e:
            logger.warning("EquityRegimeFilter.can_open failed open: %s", e)
            return True, None

    def adjust_position_size(self, size) -> float:
        try:
            sz = float(size)
        except Exception:
            return 0.0
        if sz <= 0:
            return 0.0
        if not self.enabled:
            return sz

        try:
            bearish = self._is_bearish()
            if bearish is True and self.mode != "block":
                factor = max(0.0, min(1.0, float(self.reduction_frac)))
                adj = sz * factor
                logger.info("Equity regime bearish: adjusting position size %.8f -> %.8f", sz, adj)
                return float(adj)
            return float(sz)
        except Exception as e:
            logger.warning("EquityRegimeFilter.adjust_position_size failed open: %s", e)
            return float(sz)