"""
strategies/volatility_breakout_strategy.py — Estrategia de breakout de volatilidad.

Envuelve VolatilityBreakoutStrategy de strategies_multi como BaseStrategy.
"""
from __future__ import annotations

import logging

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):
    """Señal BUY cuando el precio rompe en expansión de volatilidad.

    Condiciones (vela cerrada iloc[-2]):
      • ATR actual > ATR mínimo reciente * multiplicador (expansión de rango)
      • BB width expandiendo (salida de compresión)
      • Precio rompe banda superior con momentum
      • Régimen != BEAR
    """

    strategy_id = "vol_breakout"
    weight: float = 1.0

    def __init__(self) -> None:
        from strategies_multi import VolatilityBreakoutStrategy as _VolatilityBreakoutStrategy
        self._inner = _VolatilityBreakoutStrategy()
        self.eligible_symbols = list(self._inner.ELIGIBLE_SYMBOLS)

    def generate_signal(self, state: MarketState) -> Signal:
        try:
            sig = self._inner.signal(state.df, state.symbol, state.regime, state.balance)
        except Exception as e:
            logger.warning("[%s][vol_breakout] signal falló: %s", state.symbol, e)
            return Signal.hold(state.symbol, self.strategy_id, state.regime, f"error:{e}")

        if not sig.should_enter:
            reason = sig.meta.get("reason", "no_entry") if sig.meta else "no_entry"
            return Signal.hold(state.symbol, self.strategy_id, state.regime, reason)

        size = float(sig.position_size_frac) if sig.position_size_frac else 0.05
        return Signal(
            symbol=state.symbol,
            side="BUY",
            size_frac=size,
            strategy_id=self.strategy_id,
            confidence=0.70,
            regime=state.regime,
            meta=sig.meta or {},
        )
