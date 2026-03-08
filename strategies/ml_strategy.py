"""
strategies/ml_strategy.py — Estrategia ML (BotV5StrategyAdapter).

Envuelve BotV5StrategyAdapter para exponerlo como BaseStrategy del nuevo
sistema multi-estrategia, traduciendo EntrySignal → Signal.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "artifacts/model_momentum_v4_phase2.joblib"


class MLStrategy(BaseStrategy):
    """Estrategia basada en modelo ML (gradient boosting) + indicadores técnicos.

    Requiere:
      • close > ema200
      • ema50  > ema200
      • adx    > 25
      • ml_probability > threshold

    Bloquea en régimen BEAR.
    """

    strategy_id = "ml_momentum"
    eligible_symbols: list[str] = []   # aplica a todos los símbolos configurados
    weight: float = 2.0                # mayor peso: es el modelo principal

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_override: Optional[float] = None,
        position_size_frac: float = 0.03,
    ) -> None:
        self.model_path = model_path or (os.getenv("V5_MODEL_PATH") or "").strip() or _DEFAULT_MODEL_PATH
        thr_str = (os.getenv("V5_THRESHOLD_OVERRIDE") or "").strip()
        self.threshold_override = threshold_override if threshold_override is not None else (
            float(thr_str) if thr_str else None
        )
        self.position_size_frac = position_size_frac
        self._adapter: Any = None

    def _get_adapter(self):
        """Lazy-init del adaptador de estrategia (evita importar joblib en módulo top-level)."""
        if self._adapter is None:
            from estrategia_v5 import BotV5StrategyAdapter
            self._adapter = BotV5StrategyAdapter(
                model_path=self.model_path,
                threshold_override=self.threshold_override,
                position_size_frac=self.position_size_frac,
            )
        return self._adapter

    def generate_signal(self, state: MarketState) -> Signal:
        if state.regime == "BEAR":
            return Signal.hold(state.symbol, self.strategy_id, state.regime, "bear_regime")

        try:
            from backtesting.bt_types import StrategyContext
            from datetime import datetime, timezone

            adapter = self._get_adapter()
            indicators = state.indicators

            ctx = StrategyContext(
                symbol=state.symbol,
                i=len(state.df) - 2,
                timestamp=datetime.now(tz=timezone.utc),
                indicators=indicators,
                regime=state.regime,
                cash=state.balance,
                equity=state.equity,
                positions_open_symbol=len(state.open_positions),
                last_entry_time=None,
            )

            entry_sig = adapter.generate_entry(ctx)

        except Exception as e:
            logger.warning("[%s][ml] generate_entry falló: %s", state.symbol, e)
            return Signal.hold(state.symbol, self.strategy_id, state.regime, f"error:{e}")

        ml_meta = dict(entry_sig.meta or {})
        # Siempre extraer la probabilidad raw — incluso en HOLD — para que
        # PortfolioManager pueda usarla como escalador de posición (modo híbrido).
        raw_prob = float(ml_meta.get("ml_prob", 0.0))

        if not entry_sig.should_enter:
            # Devolver HOLD pero con raw_prob en confidence para que PortfolioManager
            # pueda compararlo contra ml_min_confidence (gate del modo híbrido).
            return Signal(
                symbol=state.symbol,
                side="HOLD",
                size_frac=0.0,
                strategy_id=self.strategy_id,
                confidence=raw_prob,    # prob real, no 0.0, para el gate híbrido
                regime=state.regime,
                meta={"reason": "ml_no_entry", "raw_prob": raw_prob, **ml_meta},
            )

        confidence = raw_prob if raw_prob > 0 else 0.5
        size = float(entry_sig.position_size_frac) if entry_sig.position_size_frac else self.position_size_frac

        return Signal(
            symbol=state.symbol,
            side="BUY",
            size_frac=size,
            strategy_id=self.strategy_id,
            confidence=confidence,
            regime=state.regime,
            meta={"raw_prob": raw_prob, **ml_meta},
        )

    def prepare_indicators(self, *, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Delegado al adaptador para compatibilidad con backtesting."""
        return self._get_adapter().prepare_indicators(symbol=symbol, df=df)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict,
    ) -> tuple[float, float, dict]:
        """Delegado al adaptador para compatibilidad con run_strategy."""
        return self._get_adapter().compute_risk_levels(
            symbol=symbol,
            regime=regime,
            buy_price=buy_price,
            indicators_row=indicators_row,
        )
