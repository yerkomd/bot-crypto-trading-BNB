"""
portfolio_manager.py — Gestor de portafolio multi-estrategia.

PortfolioManager agrega señales de múltiples estrategias usando scoring
ponderado para producir una decisión final de trading.

Lógica de agregación:
  1. Separar señales BUY, SELL y HOLD.
  2. Las señales HOLD de cualquier estrategia con veto=True bloquean la operación.
  3. Calcular score neto: sum(weight * confidence) de BUYs vs SELLs.
  4. Si score_buy > threshold y no vetado → PortfolioOrder(BUY).
  5. Si score_sell > threshold → PortfolioOrder(SELL).
  6. En caso contrario → PortfolioOrder(HOLD).
  7. El tamaño final es el mínimo de los size_frac de señales BUY activas
     (conservador, igual que MultiStrategyEngine).

Variables de entorno:
  PORTFOLIO_BUY_THRESHOLD   — score mínimo para BUY (default: 1.0)
  PORTFOLIO_SELL_THRESHOLD  — score mínimo para SELL (default: 1.0)
  PORTFOLIO_VETO_ON_HOLD    — 'false' para deshabilitar veto (default: true)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PortfolioOrder — decisión final de trading
# ---------------------------------------------------------------------------

@dataclass
class PortfolioOrder:
    """Decisión de trading consolidada del PortfolioManager."""
    symbol: str
    side: str                          # BUY | SELL | HOLD
    size_frac: float                   # Fracción del balance (0.0 si HOLD/SELL)
    score: float                       # Score neto ponderado
    triggered_by: list[str] = field(default_factory=list)   # strategy_ids contribuyentes
    vetoed_by: list[str] = field(default_factory=list)       # strategy_ids que vetaron
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def should_enter(self) -> bool:
        return self.side == "BUY"

    @property
    def should_exit(self) -> bool:
        return self.side == "SELL"


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """Agrega señales de múltiples estrategias en una decisión de portafolio.

    Parameters
    ----------
    strategies:
        Lista de estrategias con su `weight`. Usadas para lookup de pesos;
        las señales son pasadas directamente vía `decide()`.
    buy_threshold:
        Score mínimo de BUY para generar una orden de compra.
    sell_threshold:
        Score mínimo de SELL para generar una orden de venta.
    veto_on_hold:
        Si True, cualquier señal HOLD bloquea la operación (conservador).
        Si False, el HOLD se ignora y solo cuenta el score neto.
    """

    def __init__(
        self,
        strategies: Sequence[BaseStrategy],
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
        veto_on_hold: Optional[bool] = None,
    ) -> None:
        self._weights: dict[str, float] = {
            s.strategy_id: float(getattr(s, "weight", 1.0))
            for s in strategies
        }

        def _env_float(name: str, default: float) -> float:
            try:
                v = os.getenv(name)
                return float(v) if v else default
            except Exception:
                return default

        self.buy_threshold = buy_threshold if buy_threshold is not None else _env_float("PORTFOLIO_BUY_THRESHOLD", 1.0)
        self.sell_threshold = sell_threshold if sell_threshold is not None else _env_float("PORTFOLIO_SELL_THRESHOLD", 1.0)

        if veto_on_hold is not None:
            self.veto_on_hold = veto_on_hold
        else:
            raw = (os.getenv("PORTFOLIO_VETO_ON_HOLD") or "true").strip().lower()
            self.veto_on_hold = raw not in ("false", "0", "no")

    def _weight(self, strategy_id: str) -> float:
        return self._weights.get(strategy_id, 1.0)

    def decide(self, symbol: str, signals: Sequence[Signal]) -> PortfolioOrder:
        """Agrega señales y retorna la orden de portafolio.

        Parameters
        ----------
        symbol:
            Símbolo de trading (ej. BTCUSDT).
        signals:
            Señales emitidas por todas las estrategias elegibles.

        Returns
        -------
        PortfolioOrder con side=BUY, SELL o HOLD.
        """
        if not signals:
            return PortfolioOrder(symbol=symbol, side="HOLD", size_frac=0.0, score=0.0,
                                  meta={"reason": "no_signals"})

        buys: list[Signal] = []
        sells: list[Signal] = []
        holds: list[Signal] = []

        for sig in signals:
            if sig.side == "BUY":
                buys.append(sig)
            elif sig.side == "SELL":
                sells.append(sig)
            else:
                holds.append(sig)

        # --- Veto por HOLD ---
        vetoed_by: list[str] = []
        if self.veto_on_hold and holds:
            vetoed_by = [s.strategy_id for s in holds]
            logger.debug(
                "[%s] PortfolioManager: vetado por HOLD de %s",
                symbol, vetoed_by,
            )
            return PortfolioOrder(
                symbol=symbol, side="HOLD", size_frac=0.0, score=0.0,
                vetoed_by=vetoed_by,
                meta={"reason": "hold_veto", "hold_strategies": vetoed_by},
            )

        # --- Score ponderado ---
        score_buy = sum(self._weight(s.strategy_id) * s.confidence for s in buys)
        score_sell = sum(self._weight(s.strategy_id) * s.confidence for s in sells)
        score_net = score_buy - score_sell

        logger.debug(
            "[%s] PortfolioManager: score_buy=%.3f score_sell=%.3f net=%.3f "
            "buy_thr=%.3f sell_thr=%.3f",
            symbol, score_buy, score_sell, score_net,
            self.buy_threshold, self.sell_threshold,
        )

        # --- Resolución de conflictos ---
        if score_buy >= self.buy_threshold and score_buy > score_sell:
            # Tamaño: mínimo de las size_frac > 0 de señales BUY activas
            active_sizes = [s.size_frac for s in buys if s.size_frac > 0]
            size_frac = min(active_sizes) if active_sizes else 0.03

            return PortfolioOrder(
                symbol=symbol,
                side="BUY",
                size_frac=size_frac,
                score=score_net,
                triggered_by=[s.strategy_id for s in buys],
                meta={
                    "score_buy": score_buy,
                    "score_sell": score_sell,
                    "strategies_buy": [s.strategy_id for s in buys],
                    "strategies_hold": [s.strategy_id for s in holds],
                },
            )

        if score_sell >= self.sell_threshold and score_sell > score_buy:
            return PortfolioOrder(
                symbol=symbol,
                side="SELL",
                size_frac=0.0,
                score=-score_net,
                triggered_by=[s.strategy_id for s in sells],
                meta={
                    "score_buy": score_buy,
                    "score_sell": score_sell,
                    "strategies_sell": [s.strategy_id for s in sells],
                },
            )

        return PortfolioOrder(
            symbol=symbol,
            side="HOLD",
            size_frac=0.0,
            score=score_net,
            meta={
                "reason": "below_threshold",
                "score_buy": score_buy,
                "score_sell": score_sell,
                "buy_threshold": self.buy_threshold,
            },
        )
