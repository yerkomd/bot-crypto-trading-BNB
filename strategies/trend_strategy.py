"""
strategies/trend_strategy.py — Estrategia de tendencia (Trend Following).

Envuelve TrendFollowingStrategy de strategies_multi como BaseStrategy.
"""
from __future__ import annotations

import logging

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)


class TrendStrategy(BaseStrategy):
    """Señal BUY cuando BTC/ETH están en tendencia alcista confirmada.

    Condiciones (vela cerrada iloc[-2]):
      • price > EMA_SLOW
      • EMA_FAST > EMA_SLOW
      • ADX >= ADX_MIN
      • RSI en [RSI_MIN, RSI_MAX]
      • Régimen != BEAR
    """

    strategy_id = "trend_following"
    weight: float = 1.5

    def __init__(self) -> None:
        from strategies_multi import TrendFollowingStrategy as _TrendFollowingStrategy
        self._inner = _TrendFollowingStrategy()
        self.eligible_symbols = list(self._inner.ELIGIBLE_SYMBOLS)

    def generate_signal(self, state: MarketState) -> Signal:
        try:
            sig = self._inner.signal(state.df, state.symbol, state.regime, state.balance)
        except Exception as e:
            logger.warning("[%s][trend] signal falló: %s", state.symbol, e)
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
            confidence=0.75,
            regime=state.regime,
            meta=sig.meta or {},
        )
