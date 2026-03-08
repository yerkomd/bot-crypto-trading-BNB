"""
strategies/funding_arbitrage_strategy.py — Filtro de funding rate.

Envuelve FundingArbitrageStrategy de strategies_multi como BaseStrategy.
Actúa como filtro macro: bloquea entrada si funding rate es excesivamente positivo.
"""
from __future__ import annotations

import logging

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)


class FundingArbitrageStrategy(BaseStrategy):
    """Filtro macro basado en funding rate de futuros perpetuos.

    Comportamiento (spot-safe):
      • Funding > threshold → HOLD (mercado sobrecalentado, no entrar)
      • Funding <= threshold o falla API → BUY (fail-open, no bloquea spot)
      • Régimen BEAR → HOLD
    """

    strategy_id = "funding_arb"
    weight: float = 0.5   # peso menor; es filtro, no generador de señales

    def __init__(self) -> None:
        from strategies_multi import FundingArbitrageStrategy as _FundingArbitrageStrategy
        self._inner = _FundingArbitrageStrategy()
        self.eligible_symbols = list(self._inner.ELIGIBLE_SYMBOLS)

    def generate_signal(self, state: MarketState) -> Signal:
        try:
            sig = self._inner.signal(state.df, state.symbol, state.regime, state.balance)
        except Exception as e:
            logger.warning("[%s][funding] signal falló (fail-open): %s", state.symbol, e)
            # fail-open: no bloquear trading spot ante fallo de API
            return Signal(
                symbol=state.symbol,
                side="BUY",
                size_frac=0.0,   # sin sugerencia de tamaño (filtro)
                strategy_id=self.strategy_id,
                confidence=0.5,
                regime=state.regime,
                meta={"reason": "api_error_fail_open"},
            )

        if not sig.should_enter:
            reason = sig.meta.get("reason", "funding_blocked") if sig.meta else "funding_blocked"
            return Signal.hold(state.symbol, self.strategy_id, state.regime, reason)

        # Filtro pasa: señal positiva sin tamaño (no sugiere posición, solo aprueba)
        return Signal(
            symbol=state.symbol,
            side="BUY",
            size_frac=0.0,   # funding arb no sugiere tamaño propio
            strategy_id=self.strategy_id,
            confidence=0.6,
            regime=state.regime,
            meta=sig.meta or {},
        )

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception:
            pass
