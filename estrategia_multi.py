"""
estrategia_multi.py — Adaptador de backtesting para el sistema multi-estrategia v2.

Permite usar StrategyEngine + PortfolioManager con el framework de backtesting
existente (backtesting/bt_types.py). Implementa la misma interfaz que
BotV5StrategyAdapter para ser intercambiable en backtests.

Uso en backtesting:
    from estrategia_multi import MultiStrategyBacktestAdapter
    strategy = MultiStrategyBacktestAdapter()
    # luego usar exactamente igual que BotV5StrategyAdapter
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backtesting.bt_types import EntrySignal, Position, Bar, Strategy, StrategyContext
from strategies.base_strategy import MarketState
from strategy_engine import StrategyEngine, build_default_engine
from portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class MultiStrategyBacktestAdapter(Strategy):
    """Adaptador de backtesting para el sistema multi-estrategia v2.

    Interfaz compatible con BotV5StrategyAdapter (prepare_indicators,
    generate_entry, compute_risk_levels, update_trailing).

    La fuente de indicadores y niveles de riesgo se delega al MLStrategy interno
    (que a su vez usa BotV5StrategyAdapter) para mantener consistencia con el bot en vivo.
    """

    position_size_frac: float = 0.03

    _engine: Optional[StrategyEngine] = field(default=None, repr=False)
    _portfolio: Optional[PortfolioManager] = field(default=None, repr=False)
    _ml_adapter: Any = field(default=None, repr=False)
    # Stores the full prepared df per symbol so generate_entry can slice correctly
    _prepared_dfs: dict = field(default_factory=dict, repr=False)

    def _get_engine(self) -> StrategyEngine:
        if self._engine is None:
            self._engine = build_default_engine()
        return self._engine

    def _get_portfolio(self) -> PortfolioManager:
        if self._portfolio is None:
            engine = self._get_engine()
            self._portfolio = PortfolioManager(strategies=engine.strategies)
        return self._portfolio

    def _get_ml_adapter(self):
        """Obtiene el MLStrategy para delegar prepare_indicators y compute_risk_levels."""
        if self._ml_adapter is None:
            engine = self._get_engine()
            for strat in engine.strategies:
                if strat.strategy_id == "ml_momentum":
                    self._ml_adapter = strat
                    break
        return self._ml_adapter

    def prepare_indicators(self, *, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara indicadores técnicos delegando al MLStrategy (usa BotV5StrategyAdapter)."""
        ml = self._get_ml_adapter()
        if ml is not None and hasattr(ml, "prepare_indicators"):
            result = ml.prepare_indicators(symbol=symbol, df=df)
        else:
            result = df
        # Store full df so generate_entry can provide the correct slice to technical strategies
        self._prepared_dfs[symbol] = result
        return result

    def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
        """Evalúa todas las estrategias y genera señal de entrada via PortfolioManager."""
        try:
            engine = self._get_engine()
            portfolio = self._get_portfolio()

            regime = str(ctx.regime or "LATERAL").upper()

            # Use the full prepared df sliced up to bar i so that technical strategies
            # can access df.iloc[-2] (last closed candle = bar i-1) correctly.
            full_df = self._prepared_dfs.get(ctx.symbol)
            if full_df is not None and hasattr(ctx, "i") and ctx.i is not None:
                df_slice = full_df.iloc[: ctx.i + 1]
            elif full_df is not None:
                df_slice = full_df
            else:
                # Fallback: single-row df — only MLStrategy will work correctly
                df_slice = pd.DataFrame([ctx.indicators])

            state = MarketState(
                symbol=ctx.symbol,
                df=df_slice,
                regime=regime,
                balance=float(ctx.cash or 0.0),
                equity=float(ctx.equity or 0.0),
                open_positions=[],
                indicators=dict(ctx.indicators),
            )

            signals = engine.collect(state)
            order = portfolio.decide(ctx.symbol, signals)

            if not order.should_enter:
                return EntrySignal(False, self.position_size_frac, None)

            size = float(order.size_frac) if order.size_frac > 0 else self.position_size_frac
            return EntrySignal(
                True,
                size,
                {
                    "triggered_by": order.triggered_by,
                    "score": order.score,
                    "portfolio_meta": order.meta,
                },
            )

        except Exception as e:
            logger.warning("[%s] MultiStrategyBacktestAdapter.generate_entry falló: %s", ctx.symbol, e)
            return EntrySignal(False, self.position_size_frac, None)

    def compute_risk_levels(
        self,
        *,
        symbol: str,
        regime: str,
        buy_price: float,
        indicators_row: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        """Calcula TP/SL delegando al MLStrategy (usa BotV5StrategyAdapter)."""
        ml = self._get_ml_adapter()
        if ml is not None and hasattr(ml, "compute_risk_levels"):
            return ml.compute_risk_levels(
                symbol=symbol,
                regime=regime,
                buy_price=buy_price,
                indicators_row=indicators_row,
            )
        raise ValueError("MLStrategy no disponible para compute_risk_levels")

    def update_trailing(
        self,
        *,
        symbol: str,
        position: Position,
        bar: Bar,
        indicators_row: dict[str, Any],
    ) -> None:
        """Actualiza trailing stop delegando al adaptador ML."""
        ml = self._get_ml_adapter()
        if ml is not None:
            inner = getattr(ml, "_adapter", None)
            if inner is not None and hasattr(inner, "update_trailing"):
                inner.update_trailing(
                    symbol=symbol,
                    position=position,
                    bar=bar,
                    indicators_row=indicators_row,
                )
