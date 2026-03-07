from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..bt_types import Strategy, StrategyContext, EntrySignal, Position, Bar


@dataclass(frozen=True)
class BotV3_1StrategyAdapter(Strategy):
    """Adapter that reuses the production strategy logic from `bot_trading_v3_1.py`.

    No network calls: it only uses pure functions/constants (indicators, regime, ATR levels, trailing update).
    """

    rsi_confirm_min: float | None = None
    rsi_confirm_max: float | None = None
    regime_params: dict[str, dict[str, Any]] | None = None
    atr_multipliers: dict[str, dict[str, float] | None] | None = None

    def _bot(self):
        # Lazy import so importing backtesting doesn't eagerly import the full bot module.
        import bot_trading_v3_1 as bot

        return bot

    def prepare_indicators(self, *, symbol: str, df) -> pd.DataFrame:
        bot = self._bot()

        out = df.copy()
        out = bot.cal_metrics_technig(out, 14, 10, 20)

        # Precompute regime per row (detect_market_regime only looks at the last row).
        def _row_regime(r: pd.Series) -> str:
            try:
                price = r.get("close")
                ema200 = r.get("ema200")
                adx = r.get("adx")
                if pd.isna(price) or pd.isna(ema200) or pd.isna(adx):
                    return "LATERAL"
                if float(price) > float(ema200) and float(adx) >= 25.0:
                    return "BULL"
                if float(price) < float(ema200) and float(adx) >= 25.0:
                    return "BEAR"
                return "LATERAL"
            except Exception:
                return "LATERAL"

        out["regime"] = out.apply(_row_regime, axis=1)
        return out

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        bot = self._bot()

        regime = str(ctx.regime or "LATERAL").upper()
        rp = self.regime_params
        if rp is None:
            rp = bot.REGIME_PARAMS if hasattr(bot, "REGIME_PARAMS") else {"LATERAL": bot.LATERAL}
        params = (rp.get(regime) if isinstance(rp, dict) else None) or bot.LATERAL

        cooldown_s = int(params.get("BUY_COOLDOWN", 0) or 0)
        pos_size = float(params.get("POSITION_SIZE", 0.0) or 0.0)

        # Hard block: BEAR regime => no new positions.
        if regime == "BEAR" or pos_size <= 0:
            return EntrySignal(False, 0.0, None)

        # Cooldown
        if ctx.last_entry_time is not None and cooldown_s > 0:
            if (ctx.timestamp - ctx.last_entry_time).total_seconds() < float(cooldown_s):
                return EntrySignal(False, pos_size, None)

        rsi = ctx.indicators.get("rsi")
        ema200 = ctx.indicators.get("ema200")
        ema50 = ctx.indicators.get("ema50")
        close = ctx.indicators.get("close")

        if pd.isna(rsi) or pd.isna(ema200) or pd.isna(ema50) or pd.isna(close):
            return EntrySignal(False, pos_size, None)

        try:
            rsi_f = float(rsi)
            ema200_f = float(ema200)
            ema50_f = float(ema50)
            close_f = float(close)
        except Exception:
            return EntrySignal(False, pos_size, None)

        entry_ok = (close_f > ema200_f) and (ema50_f > ema200_f)

        rsi_min = float(self.rsi_confirm_min) if self.rsi_confirm_min is not None else float(bot.RSI_CONFIRM_MIN)
        rsi_max = float(self.rsi_confirm_max) if self.rsi_confirm_max is not None else float(bot.RSI_CONFIRM_MAX)
        entry_ok = entry_ok and (rsi_min <= rsi_f <= rsi_max)

        return EntrySignal(bool(entry_ok), float(pos_size), None)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        bot = self._bot()

        atr = indicators_row.get("atr")
        if atr is None or pd.isna(atr):
            raise ValueError("ATR not available")
        atr_f = float(atr)
        if atr_f <= 0:
            raise ValueError(f"ATR invalid: {atr_f}")

        r = str(regime or "LATERAL").upper()
        am = self.atr_multipliers
        if am is None:
            am = bot.ATR_MULTIPLIERS if hasattr(bot, "ATR_MULTIPLIERS") else {"LATERAL": {"tp": 2.0, "sl": 1.2, "trailing_sl": 1.0}}
        mults = (am.get(r) if isinstance(am, dict) else None) or am.get("LATERAL")
        if not mults:
            # If BEAR is None, fallback to LATERAL (production bot does not open entries in BEAR anyway).
            mults = am.get("LATERAL")
        if not mults:
            raise ValueError(f"ATR multipliers missing for regime={r}")
        tp_mult = float(mults["tp"])
        sl_mult = float(mults["sl"])
        trailing_sl_mult = float(mults["trailing_sl"])

        tp, sl = bot.calculate_atr_levels(float(buy_price), float(atr_f), float(tp_mult), float(sl_mult))

        extra = {
            "tp_initial": float(tp),
            "trailing_active": False,
            "max_price": float(buy_price),
            "atr_entry": float(atr_f),
            "tp_atr_mult": float(tp_mult),
            "sl_atr_mult": float(sl_mult),
            "trailing_sl_atr_mult": float(trailing_sl_mult),
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

        # Activate trailing after tp_initial is reached.
        if (not position.trailing_active) and float(bar.close) >= float(position.tp_initial):
            position.trailing_active = True

        if not position.trailing_active:
            return

        atr = indicators_row.get("atr")
        if atr is None or pd.isna(atr):
            return
        try:
            atr_f = float(atr)
        except Exception:
            return
        if atr_f <= 0:
            return

        trailing_mult = position.trailing_sl_atr_mult
        if trailing_mult is None or float(trailing_mult) <= 0:
            r = str(position.regime or "LATERAL").upper()
            am = self.atr_multipliers
            if am is None:
                am = bot.ATR_MULTIPLIERS if hasattr(bot, "ATR_MULTIPLIERS") else {"LATERAL": {"tp": 2.0, "sl": 1.2, "trailing_sl": 1.0}}
            mults = (am.get(r) if isinstance(am, dict) else None) or am.get("LATERAL")
            if not mults:
                return
            trailing_mult = float(mults["trailing_sl"])

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
