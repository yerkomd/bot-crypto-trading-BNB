"""
strategy_engine.py — Motor de estrategias múltiples.

StrategyEngine recopila señales de todas las estrategias elegibles para un
símbolo dado, manejando errores de forma aislada por estrategia (fail-safe).

Uso:
    engine = StrategyEngine(strategies=[MLStrategy(), TrendStrategy(), ...])
    signals = engine.collect(state)   # list[Signal]
"""
from __future__ import annotations

import logging
from typing import Sequence

from strategies.base_strategy import BaseStrategy, MarketState, Signal

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Orquestador de estrategias: evalúa cada estrategia elegible de forma independiente.

    No aplica consenso ni lógica de portafolio — eso es responsabilidad del PortfolioManager.
    """

    def __init__(self, strategies: Sequence[BaseStrategy]) -> None:
        self.strategies: list[BaseStrategy] = list(strategies)

    def collect(self, state: MarketState) -> list[Signal]:
        """Evalúa todas las estrategias elegibles y retorna sus señales.

        Cada estrategia se evalúa de forma independiente. Los fallos se loguean
        y se ignoran (fail-safe), nunca propagan excepción al caller.

        Returns:
            Lista de Signal (puede incluir señales HOLD). Vacía si ninguna estrategia
            aplica al símbolo o todas fallan.
        """
        signals: list[Signal] = []

        for strat in self.strategies:
            if not strat.is_eligible(state.symbol):
                logger.debug("[%s] %s: no elegible, skip", state.symbol, strat.strategy_id)
                continue

            try:
                sig = strat.generate_signal(state)
                signals.append(sig)
                logger.debug(
                    "[%s] %s: side=%s size=%.3f conf=%.2f",
                    state.symbol, strat.strategy_id,
                    sig.side, sig.size_frac, sig.confidence,
                )
            except Exception as e:
                logger.warning(
                    "[%s] %s: generate_signal lanzó excepción (ignorado): %s",
                    state.symbol, strat.strategy_id, e,
                )

        return signals

    def close(self) -> None:
        """Libera recursos de todas las estrategias."""
        for strat in self.strategies:
            try:
                strat.close()
            except Exception as e:
                logger.debug("close() en %s falló: %s", strat.strategy_id, e)


def build_default_engine() -> StrategyEngine:
    """Construye el motor por defecto con las 5 estrategias activas.

    Estrategias incluidas:
      1. MLStrategy        — modelo ML + técnicos (peso 2.0)
      2. TrendStrategy     — trend following BTC/ETH (peso 1.5)
      3. MeanReversionStrategy — reversión altcoins (peso 1.0)
      4. FundingArbitrageStrategy — filtro funding (peso 0.5)
      5. VolatilityBreakoutStrategy — breakout volatilidad (peso 1.0)
    """
    from strategies.ml_strategy import MLStrategy
    from strategies.trend_strategy import TrendStrategy
    from strategies.mean_reversion_strategy import MeanReversionStrategy
    from strategies.funding_arbitrage_strategy import FundingArbitrageStrategy
    from strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy

    return StrategyEngine(strategies=[
        MLStrategy(),
        TrendStrategy(),
        MeanReversionStrategy(),
        FundingArbitrageStrategy(),
        VolatilityBreakoutStrategy(),
    ])
