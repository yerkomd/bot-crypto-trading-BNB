from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..bt_types import Strategy, StrategyContext, EntrySignal, Position, Bar


@dataclass(frozen=True)
class BotV2_2StrategyAdapter(Strategy):
    """Adapter that reuses the production strategy logic from `bot_trading_v2_2.py`.

    No network calls: it only uses indicators/params and simulates the entry logic.

    Semantics implemented:
    - Regime computed similarly to the bot (EMA200 + ADX threshold).
    - Main entry: RSI < regime RSI_THRESHOLD AND StochRSI K > D, with a per-regime cooldown.
    - DCA entry: 5-bar change <= -5%, with a per-regime cooldown.

    Important: the backtest engine exits ONLY via stop_loss; TP is used as a reference for
    when to update trailing (engine doesn't fill on TP).
    """

    regime_params: dict[str, dict[str, Any]] | None = None

    def _bot(self):
        import bot_trading_v2_2 as bot

        return bot

    def prepare_indicators(self, *, symbol: str, df) -> pd.DataFrame:
        bot = self._bot()

        out = df.copy()
        out = bot.cal_metrics_technig(out, 14, 10, 20)

        # Precompute regime per row.
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

        # Precompute 5-bar percent change used for DCA trigger.
        try:
            out["change_5"] = (out["close"] / out["close"].shift(5) - 1.0) * 100.0
        except Exception:
            out["change_5"] = pd.NA

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

        # Cooldown
        if ctx.last_entry_time is not None and cooldown_s > 0:
            if (ctx.timestamp - ctx.last_entry_time).total_seconds() < float(cooldown_s):
                return EntrySignal(False, pos_size, None)

        if pos_size <= 0:
            return EntrySignal(False, 0.0, None)

        rsi = ctx.indicators.get("rsi")
        k = ctx.indicators.get("stochrsi_k")
        d = ctx.indicators.get("stochrsi_d")
        ch5 = ctx.indicators.get("change_5")

        if rsi is None or k is None or d is None:
            return EntrySignal(False, pos_size, None)

        if pd.isna(rsi) or pd.isna(k) or pd.isna(d):
            return EntrySignal(False, pos_size, None)

        try:
            rsi_f = float(rsi)
            k_f = float(k)
            d_f = float(d)
        except Exception:
            return EntrySignal(False, pos_size, None)

        rsi_threshold = float(params.get("RSI_THRESHOLD", getattr(bot, "rsi_threshold", 30.0)) or 30.0)

        main_ok = (rsi_f < rsi_threshold) and (k_f > d_f) and (int(ctx.positions_open_symbol) < 5)

        dca_ok = False
        if ch5 is not None and not pd.isna(ch5):
            try:
                ch5_f = float(ch5)
                dca_ok = (ch5_f <= -5.0) and (int(ctx.positions_open_symbol) < 9)
            except Exception:
                dca_ok = False

        return EntrySignal(bool(main_ok or dca_ok), float(pos_size), None)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        bot = self._bot()

        r = str(regime or "LATERAL").upper()
        rp = self.regime_params
        if rp is None:
            rp = bot.REGIME_PARAMS if hasattr(bot, "REGIME_PARAMS") else {"LATERAL": bot.LATERAL}
        params = (rp.get(r) if isinstance(rp, dict) else None) or bot.LATERAL

        tp_pct = float(params.get("TAKE_PROFIT_PCT", getattr(bot, "take_profit_pct", 2.0)) or 2.0)
        sl_pct = float(params.get("STOP_LOSS_PCT", getattr(bot, "stop_loss_pct", 1.0)) or 1.0)
        trailing_tp_pct = float(params.get("TRAILING_TP_PCT", getattr(bot, "trailing_take_profit_pct", 0.5)) or 0.5)
        trailing_sl_pct = float(params.get("TRAILING_SL_PCT", getattr(bot, "trailing_stop_pct", 0.5)) or 0.5)

        tp = float(buy_price) * (1.0 + float(tp_pct) / 100.0)
        sl = float(buy_price) * (1.0 - float(sl_pct) / 100.0)

        extra = {
            "tp_initial": float(tp),
            "trailing_active": False,
            "max_price": float(buy_price),
            # Store trailing pct values in these fields for adapter-only use.
            "tp_atr_mult": float(trailing_tp_pct),
            "trailing_sl_atr_mult": float(trailing_sl_pct),
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
        # v2.2 trailing semantics (simplified to what engine supports):
        # - when close >= take_profit: set new take_profit and raise stop_loss using trailing pct
        trailing_tp_pct = float(position.tp_atr_mult or 0.0)
        trailing_sl_pct = float(position.trailing_sl_atr_mult or 0.0)

        if trailing_tp_pct <= 0 or trailing_sl_pct <= 0:
            return

        if float(bar.close) >= float(position.take_profit):
            position.trailing_active = True
            new_tp = float(bar.close) * (1.0 + trailing_tp_pct / 100.0)
            candidate_sl = float(bar.close) * (1.0 - trailing_sl_pct / 100.0)

            # Never decrease the stop.
            if candidate_sl > float(position.stop_loss):
                position.stop_loss = float(candidate_sl)
            position.take_profit = float(new_tp)

            # Track max observed price
            if float(bar.close) > float(position.max_price):
                position.max_price = float(bar.close)
