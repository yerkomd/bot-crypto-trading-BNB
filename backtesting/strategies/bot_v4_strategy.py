from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..bt_types import Strategy, StrategyContext, EntrySignal, Position, Bar


@dataclass(frozen=True)
class BotV4StrategyAdapter(Strategy):
    """Adapter for the production `bot_trading_v4.py` strategy logic.

    Notes:
    - No network calls.
    - Backtest engine enters at bar i OPEN, using signals from bar i-1.
    - v4 strategy uses: close > EMA200 and breakout above highest high of last 10 bars (excluding signal bar).
    - SL is ATR-based: SL = entry_price - 1.5 * ATR
    - Trailing is ATR-based and active immediately: 1.0 * ATR

    The backtesting engine exits only via stop_loss, so take_profit is a placeholder.
    """

    position_size_frac: float = 0.03
    breakout_window: int = 10
    sl_atr_mult: float = 1.5
    trailing_sl_atr_mult: float = 1.0

    def _bot(self):
        # Lazy import to avoid importing full bot module unless used.
        import bot_trading_v4 as bot

        return bot

    def prepare_indicators(self, *, symbol: str, df) -> pd.DataFrame:
        bot = self._bot()

        out = df.copy()
        # Reuse the bot's indicator calculator (keeps ATR/EMA definitions consistent).
        out = bot.cal_metrics_technig(out, 14, 10, 20)

        # Ensure required columns exist.
        if "ema200" not in out.columns:
            try:
                out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=1).mean()
            except Exception:
                out["ema200"] = pd.NA

        if "atr" not in out.columns:
            out["atr"] = pd.NA

        # Breakout reference: highest high of last N bars, excluding the signal bar.
        # Engine uses row i-1 for signal; so for row r this must be max(high[r-N .. r-1]).
        out["highest_high_10"] = out["high"].rolling(window=int(self.breakout_window)).max().shift(1)

        # Engine expects a regime field (even if unused by this strategy).
        out["regime"] = "LATERAL"

        return out

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        # Max 1 position per symbol for v4.
        if int(ctx.positions_open_symbol) >= 1:
            return EntrySignal(False, float(self.position_size_frac), None)

        close = ctx.indicators.get("close")
        ema200 = ctx.indicators.get("ema200")
        hh10 = ctx.indicators.get("highest_high_10")

        if close is None or ema200 is None or hh10 is None:
            return EntrySignal(False, float(self.position_size_frac), None)

        if pd.isna(close) or pd.isna(ema200) or pd.isna(hh10):
            return EntrySignal(False, float(self.position_size_frac), None)

        try:
            close_f = float(close)
            ema200_f = float(ema200)
            hh10_f = float(hh10)
        except Exception:
            return EntrySignal(False, float(self.position_size_frac), None)

        entry_ok = (close_f > ema200_f) and (close_f > hh10_f)
        return EntrySignal(bool(entry_ok), float(self.position_size_frac), None)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        atr = indicators_row.get("atr")
        if atr is None or pd.isna(atr):
            raise ValueError("ATR not available")

        atr_f = float(atr)
        if atr_f <= 0:
            raise ValueError(f"ATR invalid: {atr_f}")

        sl = float(buy_price) - float(self.sl_atr_mult) * float(atr_f)

        # Placeholder TP: not used for direct exits in engine; kept for schema compatibility.
        tp = float(buy_price)

        extra = {
            "tp_initial": float(tp),
            "trailing_active": True,
            "max_price": float(buy_price),
            "atr_entry": float(atr_f),
            "sl_atr_mult": float(self.sl_atr_mult),
            "trailing_sl_atr_mult": float(self.trailing_sl_atr_mult),
        }
        return float(tp), float(sl), extra

    def update_trailing(
        self,
        *,
        symbol: str,
        position: Position,
        bar: Bar,
        indicators_row: dict[str, Any],
    ) -> None:
        bot = self._bot()

        atr = indicators_row.get("atr")
        if atr is None or pd.isna(atr):
            return

        try:
            atr_f = float(atr)
        except Exception:
            return

        if atr_f <= 0:
            return

        position.trailing_active = True

        trailing_mult = float(position.trailing_sl_atr_mult or 0.0)
        if trailing_mult <= 0:
            trailing_mult = float(self.trailing_sl_atr_mult)

        tmp = {
            "trailing_active": True,
            "buy_price": float(position.entry_price),
            "max_price": float(position.max_price),
            "stop_loss": float(position.stop_loss),
        }

        changed = bot.update_trailing_stop_atr(tmp, float(bar.close), float(atr_f), float(trailing_mult))
        if changed:
            position.max_price = float(tmp.get("max_price", position.max_price))
            position.stop_loss = float(tmp.get("stop_loss", position.stop_loss))
