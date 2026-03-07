from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, Any


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass
class Position:
    symbol: str
    qty: float
    entry_time: datetime
    entry_index: int
    entry_price: float
    regime: str

    take_profit: float
    stop_loss: float
    tp_initial: float
    trailing_active: bool
    max_price: float

    entry_fee: float = 0.0

    atr_entry: Optional[float] = None
    tp_atr_mult: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    trailing_sl_atr_mult: Optional[float] = None


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float
    reason: str
    pnl: float
    pnl_pct: float
    fees_paid: float
    duration_bars: int


@dataclass(frozen=True)
class StrategyContext:
    symbol: str
    i: int
    # signal is computed from bar i-1, entry would be at bar i open
    timestamp: datetime
    indicators: dict[str, Any]
    regime: str
    cash: float
    equity: float
    positions_open_symbol: int
    last_entry_time: Optional[datetime]


@dataclass(frozen=True)
class EntrySignal:
    should_enter: bool
    position_size_frac: float
    # optional override; if None the engine will compute risk levels from ATR+regime
    meta: dict[str, Any] | None = None


class Strategy(Protocol):
    """Strategy interface for the backtest engine."""

    def prepare_indicators(self, *, symbol: str, df) -> Any:
        """Return a DataFrame-like object with indicator columns."""

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        """Return entry decision for this symbol at time ctx.timestamp."""

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        """Return (take_profit, stop_loss, extra_meta)."""

    def update_trailing(
        self,
        *,
        symbol: str,
        position: Position,
        bar: Bar,
        indicators_row: dict[str, Any],
    ) -> None:
        """Update trailing fields in-place at the end of bar."""
