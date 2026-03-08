"""
portfolio_manager.py — Gestor de portafolio multi-estrategia.

PortfolioManager agrega señales de múltiples estrategias usando scoring
ponderado para producir una decisión final de trading.

Lógica de agregación (modo estándar):
  1. Separar señales BUY, SELL y HOLD.
  2. Las señales HOLD de cualquier estrategia con veto=True bloquean la operación.
  3. Calcular score neto: sum(weight * confidence) de BUYs vs SELLs.
  4. Si score_buy > threshold y no vetado → PortfolioOrder(BUY).
  5. Si score_sell > threshold → PortfolioOrder(SELL).
  6. En caso contrario → PortfolioOrder(HOLD).
  7. El tamaño final es el mínimo de los size_frac de señales BUY activas
     (conservador, igual que MultiStrategyEngine).

Modo híbrido (ML_HYBRID_MODE=true):
  El ML deja de ser un veto binario y pasa a escalar el tamaño de posición:
  • ml_prob < ml_min_confidence  → HOLD (gate mínimo de calidad)
  • ml_prob ∈ [min, 0.55)        → size × ML_SIZE_SCALE_LOW  (entrada mínima)
  • ml_prob ∈ [0.55, 0.70)       → size × ML_SIZE_SCALE_MID  (entrada normal)
  • ml_prob ≥ 0.70               → size × ML_SIZE_SCALE_HIGH (entrada ampliada)
  El trigger de entrada sigue siendo TREND / VOL_BREAKOUT (score >= threshold).

Variables de entorno:
  PORTFOLIO_BUY_THRESHOLD   — score mínimo para BUY (default: 1.0)
  PORTFOLIO_SELL_THRESHOLD  — score mínimo para SELL (default: 1.0)
  PORTFOLIO_VETO_ON_HOLD    — 'false' para deshabilitar veto (default: true)
  ML_HYBRID_MODE            — 'true' activa escalado por confianza ML (default: false)
  ML_HYBRID_STRATEGY_ID     — strategy_id del modelo ML (default: ml_momentum)
  ML_MIN_CONFIDENCE         — gate mínimo en modo híbrido (default: 0.40)
  ML_SIZE_SCALE_LOW         — escala para conf ∈ [min, 0.55) (default: 0.5)
  ML_SIZE_SCALE_MID         — escala para conf ∈ [0.55, 0.70) (default: 1.0)
  ML_SIZE_SCALE_HIGH        — escala para conf ≥ 0.70 (default: 1.5)
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
        # Modo híbrido: ML como escalador de posición
        ml_hybrid_mode: Optional[bool] = None,
        ml_hybrid_strategy_id: Optional[str] = None,
        ml_min_confidence: Optional[float] = None,
        ml_size_scale_low: Optional[float] = None,
        ml_size_scale_mid: Optional[float] = None,
        ml_size_scale_high: Optional[float] = None,
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

        def _env_bool(name: str, default: bool) -> bool:
            raw = (os.getenv(name) or "").strip().lower()
            if not raw:
                return default
            return raw not in ("false", "0", "no")

        self.buy_threshold = buy_threshold if buy_threshold is not None else _env_float("PORTFOLIO_BUY_THRESHOLD", 1.0)
        self.sell_threshold = sell_threshold if sell_threshold is not None else _env_float("PORTFOLIO_SELL_THRESHOLD", 1.0)

        if veto_on_hold is not None:
            self.veto_on_hold = veto_on_hold
        else:
            raw = (os.getenv("PORTFOLIO_VETO_ON_HOLD") or "true").strip().lower()
            self.veto_on_hold = raw not in ("false", "0", "no")

        # Modo híbrido
        self.ml_hybrid_mode = ml_hybrid_mode if ml_hybrid_mode is not None else _env_bool("ML_HYBRID_MODE", False)
        self.ml_hybrid_strategy_id = ml_hybrid_strategy_id or (os.getenv("ML_HYBRID_STRATEGY_ID") or "ml_momentum").strip()
        self.ml_min_confidence = ml_min_confidence if ml_min_confidence is not None else _env_float("ML_MIN_CONFIDENCE", 0.40)
        self.ml_size_scale_low = ml_size_scale_low if ml_size_scale_low is not None else _env_float("ML_SIZE_SCALE_LOW", 0.5)
        self.ml_size_scale_mid = ml_size_scale_mid if ml_size_scale_mid is not None else _env_float("ML_SIZE_SCALE_MID", 1.0)
        self.ml_size_scale_high = ml_size_scale_high if ml_size_scale_high is not None else _env_float("ML_SIZE_SCALE_HIGH", 1.5)

    def _weight(self, strategy_id: str) -> float:
        return self._weights.get(strategy_id, 1.0)

    def _ml_size_scale(self, ml_conf: float) -> float:
        """Devuelve el factor de escala de tamaño según la confianza del modelo ML."""
        if ml_conf >= 0.70:
            return self.ml_size_scale_high
        if ml_conf >= 0.55:
            return self.ml_size_scale_mid
        return self.ml_size_scale_low

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
        ml_signal: Optional[Signal] = None

        for sig in signals:
            if sig.strategy_id == self.ml_hybrid_strategy_id:
                ml_signal = sig
            if sig.side == "BUY":
                buys.append(sig)
            elif sig.side == "SELL":
                sells.append(sig)
            else:
                holds.append(sig)

        # ── Modo híbrido: ML como escalador de posición ──────────────────────
        if self.ml_hybrid_mode:
            # Extraer confianza raw del ML (disponible en HOLD.confidence gracias
            # al cambio en MLStrategy.generate_signal).
            ml_conf = ml_signal.confidence if ml_signal is not None else 0.0

            # Gate mínimo de calidad: si ML tiene muy baja confianza → no entrar
            if ml_conf < self.ml_min_confidence:
                logger.debug(
                    "[%s] PortfolioManager hybrid: ml_conf=%.3f < min=%.3f → HOLD",
                    symbol, ml_conf, self.ml_min_confidence,
                )
                return PortfolioOrder(
                    symbol=symbol, side="HOLD", size_frac=0.0, score=0.0,
                    meta={
                        "reason": "ml_below_min_confidence",
                        "ml_conf": ml_conf,
                        "ml_min_confidence": self.ml_min_confidence,
                    },
                )

            # Score ponderado (sin considerar la señal ML como parte del score de
            # entrada — su rol es solo escalar el tamaño)
            non_ml_buys = [s for s in buys if s.strategy_id != self.ml_hybrid_strategy_id]
            non_ml_sells = [s for s in sells if s.strategy_id != self.ml_hybrid_strategy_id]
            score_buy = sum(self._weight(s.strategy_id) * s.confidence for s in non_ml_buys)
            score_sell = sum(self._weight(s.strategy_id) * s.confidence for s in non_ml_sells)
            score_net = score_buy - score_sell

            logger.debug(
                "[%s] PortfolioManager hybrid: ml_conf=%.3f scale=%.2f "
                "score_buy=%.3f score_sell=%.3f buy_thr=%.3f",
                symbol, ml_conf, self._ml_size_scale(ml_conf),
                score_buy, score_sell, self.buy_threshold,
            )

            if score_buy >= self.buy_threshold and score_buy > score_sell:
                active_sizes = [s.size_frac for s in non_ml_buys if s.size_frac > 0]
                base_size = min(active_sizes) if active_sizes else 0.03
                scale = self._ml_size_scale(ml_conf)
                scaled_size = base_size * scale

                return PortfolioOrder(
                    symbol=symbol,
                    side="BUY",
                    size_frac=scaled_size,
                    score=score_net,
                    triggered_by=[s.strategy_id for s in non_ml_buys],
                    meta={
                        "mode": "hybrid",
                        "ml_conf": ml_conf,
                        "ml_size_scale": scale,
                        "base_size": base_size,
                        "score_buy": score_buy,
                        "score_sell": score_sell,
                        "strategies_buy": [s.strategy_id for s in non_ml_buys],
                    },
                )

            if score_sell >= self.sell_threshold and score_sell > score_buy:
                return PortfolioOrder(
                    symbol=symbol,
                    side="SELL",
                    size_frac=0.0,
                    score=-score_net,
                    triggered_by=[s.strategy_id for s in non_ml_sells],
                    meta={
                        "mode": "hybrid",
                        "score_buy": score_buy,
                        "score_sell": score_sell,
                    },
                )

            return PortfolioOrder(
                symbol=symbol, side="HOLD", size_frac=0.0, score=score_net,
                meta={
                    "reason": "below_threshold",
                    "mode": "hybrid",
                    "ml_conf": ml_conf,
                    "score_buy": score_buy,
                    "buy_threshold": self.buy_threshold,
                },
            )

        # ── Modo estándar ────────────────────────────────────────────────────

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
