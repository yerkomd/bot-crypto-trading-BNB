"""
strategies/ — Paquete de estrategias del sistema multi-estrategia.

Exporta los tipos base y todas las implementaciones de estrategia.
"""
from strategies.base_strategy import BaseStrategy, MarketState, Signal
from strategies.ml_strategy import MLStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.funding_arbitrage_strategy import FundingArbitrageStrategy
from strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy

__all__ = [
    "BaseStrategy",
    "MarketState",
    "Signal",
    "MLStrategy",
    "TrendStrategy",
    "MeanReversionStrategy",
    "FundingArbitrageStrategy",
    "VolatilityBreakoutStrategy",
]
