"""
strategies/mean_reversion_strategy.py — Estrategia de reversión a la media.

Envuelve MeanReversionStrategy de strategies_multi como BaseStrategy.
"""
from __future__ import annotations

import logging

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Señal BUY cuando altcoins están sobrevendidas y convergen hacia la media.

    Condiciones (vela cerrada iloc[-2]):
      • price < BB_LBAND
      • RSI <= RSI_OVERSOLD
      • RSI cruzando hacia arriba (confirmación de giro)
      • Régimen != BEAR
    """

    strategy_id = "mean_reversion"
    weight: float = 1.0

    def __init__(self) -> None:
        from strategies_multi import MeanReversionStrategy as _MeanReversionStrategy
        self._inner = _MeanReversionStrategy()
        self.eligible_symbols = list(self._inner.ELIGIBLE_SYMBOLS)

    def generate_signal(self, state: MarketState) -> Signal:
        try:
            sig = self._inner.signal(state.df, state.symbol, state.regime, state.balance)
        except Exception as e:
            logger.warning("[%s][mean_rev] signal falló: %s", state.symbol, e)
            return Signal.hold(state.symbol, self.strategy_id, state.regime, f"error:{e}")

        if not sig.should_enter:
            reason = sig.meta.get("reason", "no_entry") if sig.meta else "no_entry"
            return Signal.hold(state.symbol, self.strategy_id, state.regime, reason)

        size = float(sig.position_size_frac) if sig.position_size_frac else 0.04
        return Signal(
            symbol=state.symbol,
            side="BUY",
            size_frac=size,
            strategy_id=self.strategy_id,
            confidence=0.65,
            regime=state.regime,
            meta=sig.meta or {},
        )
