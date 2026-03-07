from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from backtesting.bt_types import Strategy, StrategyContext, EntrySignal, Position, Bar


@dataclass
class BotV5StrategyAdapter(Strategy):
    """Backtest adapter for bot_trading_v5 (no network calls).

    Entry conditions (4 technical vars + ML):
    - close > ema200
    - ema50 > ema200
    - adx > 25
    - ml_probability > threshold

    Hard block:
    - BEAR regime => no new positions

    Risk levels (ATR-based):
    - TP = 2.5 * ATR
    - SL = 1.5 * ATR
    - trailing SL = 1.2 * ATR
    """

    model_path: str = "artifacts/model_momentum_v4_phase2.joblib"
    model_loader: Callable[[str], Any] | None = None
    threshold_override: float | None = None
    position_size_frac: float = 0.03

    _model_bundle: dict[str, Any] | None = None
    _model_error: str | None = None

    def _bot(self):
        import bot_trading_v5 as bot

        return bot

    def _load_bundle(self) -> dict[str, Any] | None:
        if self._model_bundle is not None or self._model_error is not None:
            return self._model_bundle

        loader = self.model_loader
        if loader is None:
            import joblib

            loader = joblib.load

        try:
            obj = loader(str(self.model_path))
        except Exception as e:
            self._model_error = f"{type(e).__name__}: {e}"
            return None

        if isinstance(obj, dict):
            bundle = dict(obj)
            if bundle.get("model") is None and isinstance(bundle.get("production_model"), dict):
                bundle["model"] = bundle["production_model"].get("model")
        else:
            bundle = {"model": obj}

        model = bundle.get("model")

        # Metadata defaults
        bundle.setdefault("signals_enabled", True)
        bundle.setdefault("operating_mode", "active")
        bundle.setdefault("operating_threshold", 0.60)

        features = bundle.get("features")
        if features is None:
            try:
                fin = getattr(model, "feature_names_in_", None)
                if fin is not None:
                    features = list(fin)
            except Exception:
                features = None
        if features is None:
            features = ["close", "ema200", "ema50", "adx"]
        bundle["features"] = list(features)

        self._model_bundle = bundle
        return bundle

    def prepare_indicators(self, *, symbol: str, df) -> pd.DataFrame:
        bot = self._bot()

        out = df.copy()
        out = bot.cal_metrics_technig(out, 14, 10, 20)

        # Defensive: ensure required columns exist (tests expect them).
        if "ema200" not in out.columns:
            out["ema200"] = bot.ta.trend.EMAIndicator(out["close"], window=200).ema_indicator()
        if "ema50" not in out.columns:
            out["ema50"] = bot.ta.trend.EMAIndicator(out["close"], window=50).ema_indicator()
        if "atr" not in out.columns:
            atr_ind = bot.ta.volatility.AverageTrueRange(
                high=out["high"], low=out["low"], close=out["close"], window=int(getattr(bot, "ATR_WINDOW", 14))
            )
            out["atr"] = atr_ind.average_true_range()
        if "adx" not in out.columns:
            try:
                adx_ind = bot.ta.trend.ADXIndicator(high=out["high"], low=out["low"], close=out["close"], window=14)
                out["adx"] = adx_ind.adx()
            except Exception:
                out["adx"] = pd.NA

        # Compute ML engineered features if the artifact requests them and they are OHLCV-derivable.
        try:
            b = self._load_bundle()
            feats = list((b or {}).get("features") or [])
            if feats:
                out = bot._ensure_ml_features(out, feats)
        except Exception:
            pass

        # Regime per row (avoid lookahead; detect_market_regime reads last row only).
        def _row_regime(r: pd.Series) -> str:
            try:
                close = r.get("close")
                ema200 = r.get("ema200")
                adx = r.get("adx")
                if pd.isna(close) or pd.isna(ema200) or pd.isna(adx):
                    return "LATERAL"
                close_f = float(close)
                ema200_f = float(ema200)
                adx_f = float(adx)
                if adx_f >= 25.0 and close_f > ema200_f:
                    return "BULL"
                if adx_f >= 25.0 and close_f < ema200_f:
                    return "BEAR"
                return "LATERAL"
            except Exception:
                return "LATERAL"

        out["regime"] = out.apply(_row_regime, axis=1)
        return out

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        regime = str(ctx.regime or "LATERAL").upper()
        pos_size = float(self.position_size_frac)

        # Hard block: BEAR regime => no new positions.
        if regime == "BEAR" or pos_size <= 0:
            return EntrySignal(False, 0.0, None)

        close = ctx.indicators.get("close")
        ema200 = ctx.indicators.get("ema200")
        ema50 = ctx.indicators.get("ema50")
        adx = ctx.indicators.get("adx")

        if pd.isna(close) or pd.isna(ema200) or pd.isna(ema50) or pd.isna(adx):
            return EntrySignal(False, pos_size, None)

        try:
            close_f = float(close)
            ema200_f = float(ema200)
            ema50_f = float(ema50)
            adx_f = float(adx)
        except Exception:
            return EntrySignal(False, pos_size, None)

        tech_ok = (close_f > ema200_f) and (ema50_f > ema200_f) and (adx_f > 25.0)
        if not tech_ok:
            return EntrySignal(False, pos_size, None)

        bundle = self._load_bundle()
        if not bundle:
            return EntrySignal(False, pos_size, None)

        if not bool(bundle.get("signals_enabled", True)):
            return EntrySignal(False, pos_size, None)

        if str(bundle.get("operating_mode", "active")).strip().lower() != "active":
            # shadow mode => never enter
            return EntrySignal(False, pos_size, None)

        model = bundle.get("model")
        if model is None or not hasattr(model, "predict_proba"):
            return EntrySignal(False, pos_size, None)

        features = list(bundle.get("features") or [])
        if not features:
            return EntrySignal(False, pos_size, None)

        missing = [f for f in features if f not in ctx.indicators or pd.isna(ctx.indicators.get(f))]
        if missing:
            return EntrySignal(False, pos_size, None)

        try:
            row = pd.DataFrame([{f: float(ctx.indicators.get(f)) for f in features}], columns=features)
            proba = model.predict_proba(row)
            p = float(proba[0][1]) if len(proba[0]) >= 2 else float(proba[0][0])
        except Exception:
            return EntrySignal(False, pos_size, None)

        thr = float(self.threshold_override) if self.threshold_override is not None else float(bundle.get("operating_threshold") or 0.60)
        should_enter = bool(p > thr)
        meta = {"ml_prob": float(p), "ml_threshold": float(thr)}
        return EntrySignal(should_enter, pos_size, meta)

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

        tp_mult = 2.5
        sl_mult = 1.5
        trailing_mult = 1.2

        tp, sl = bot.calculate_atr_levels(float(buy_price), float(atr_f), float(tp_mult), float(sl_mult))

        extra = {
            "tp_initial": float(tp),
            "trailing_active": False,
            "max_price": float(buy_price),
            "atr_entry": float(atr_f),
            "tp_atr_mult": float(tp_mult),
            "sl_atr_mult": float(sl_mult),
            "trailing_sl_atr_mult": float(trailing_mult),
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
            trailing_mult = 1.2

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
