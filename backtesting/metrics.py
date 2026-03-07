from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Metrics:
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown: float
    profit_factor: Optional[float]
    expectancy: Optional[float]
    calmar: Optional[float]
    win_rate: Optional[float]
    avg_trade_duration_s: Optional[float]


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / np.where(peaks == 0, 1.0, peaks)
    return float(np.max(dd))


def _infer_periods_per_year(timestamps: list[datetime]) -> Optional[float]:
    if len(timestamps) < 3:
        return None
    deltas = np.diff(np.array([t.timestamp() for t in timestamps], dtype=float))
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return None
    median_s = float(np.median(deltas))
    if median_s <= 0:
        return None
    return float((365.0 * 24.0 * 3600.0) / median_s)


def sharpe_ratio(returns: np.ndarray, *, periods_per_year: float) -> Optional[float]:
    if returns.size < 2:
        return None
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return None
    return float((mu / sd) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, *, periods_per_year: float) -> Optional[float]:
    if returns.size < 2:
        return None
    mu = float(np.mean(returns))
    downside = returns[returns < 0]
    if downside.size < 2:
        return None
    dd = float(np.std(downside, ddof=1))
    if dd == 0:
        return None
    return float((mu / dd) * np.sqrt(periods_per_year))


def profit_factor(trade_pnls: np.ndarray) -> Optional[float]:
    if trade_pnls.size == 0:
        return None
    gross_profit = float(np.sum(trade_pnls[trade_pnls > 0]))
    gross_loss = float(np.sum(trade_pnls[trade_pnls < 0]))
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return float(gross_profit / abs(gross_loss))


def expectancy(trade_pnls: np.ndarray) -> Optional[float]:
    if trade_pnls.size == 0:
        return None
    return float(np.mean(trade_pnls))


def calmar_ratio(
    *,
    equity_start: float,
    equity_end: float,
    max_dd: float,
    timestamps: list[datetime],
) -> Optional[float]:
    if equity_start <= 0:
        return None
    if max_dd <= 0:
        return None
    if len(timestamps) < 2:
        return None
    days = (timestamps[-1] - timestamps[0]).total_seconds() / (24.0 * 3600.0)
    if days <= 0:
        return None
    years = days / 365.0
    if years <= 0:
        return None
    cagr = (equity_end / equity_start) ** (1.0 / years) - 1.0
    return float(cagr / max_dd)


def win_rate(trade_pnls: np.ndarray) -> Optional[float]:
    if trade_pnls.size == 0:
        return None
    return float(np.mean(trade_pnls > 0))


def avg_trade_duration_s(entry_ts: list[datetime], exit_ts: list[datetime]) -> Optional[float]:
    if not entry_ts or not exit_ts or len(entry_ts) != len(exit_ts):
        return None
    durs = [(b - a).total_seconds() for a, b in zip(entry_ts, exit_ts)]
    durs = [d for d in durs if d >= 0]
    if not durs:
        return None
    return float(np.mean(durs))


def compute_metrics(
    *,
    timestamps: list[datetime],
    equity: list[float],
    trade_pnls: list[float],
    trade_entry_ts: list[datetime],
    trade_exit_ts: list[datetime],
) -> Metrics:
    equity_arr = np.array(equity, dtype=float)
    rets = np.diff(equity_arr) / np.where(equity_arr[:-1] == 0, 1.0, equity_arr[:-1])
    ppy = _infer_periods_per_year(timestamps) or 365.0

    mdd = max_drawdown(equity_arr)
    tp = np.array(trade_pnls, dtype=float)

    return Metrics(
        sharpe=sharpe_ratio(rets, periods_per_year=float(ppy)),
        sortino=sortino_ratio(rets, periods_per_year=float(ppy)),
        max_drawdown=float(mdd),
        profit_factor=profit_factor(tp),
        expectancy=expectancy(tp),
        calmar=calmar_ratio(
            equity_start=float(equity_arr[0]) if equity_arr.size else 0.0,
            equity_end=float(equity_arr[-1]) if equity_arr.size else 0.0,
            max_dd=float(mdd),
            timestamps=timestamps,
        ),
        win_rate=win_rate(tp),
        avg_trade_duration_s=avg_trade_duration_s(trade_entry_ts, trade_exit_ts),
    )
