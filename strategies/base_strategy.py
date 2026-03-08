"""
strategies/base_strategy.py — Interfaz base para todas las estrategias.

Define los tipos de datos compartidos:
  • MarketState — snapshot del mercado para un símbolo en un instante dado
  • Signal      — señal de trading generada por una estrategia
  • BaseStrategy — clase base abstracta con generate_signal()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# MarketState — snapshot del mercado para un símbolo
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Snapshot de mercado pasado a cada estrategia para evaluar señales."""
    symbol: str
    df: pd.DataFrame                   # OHLCV + indicadores calculados
    regime: str                        # BULL | BEAR | LATERAL
    balance: float                     # USDT disponible
    equity: float                      # Equity total estimada
    open_positions: list[dict]         # Posiciones abiertas para este símbolo
    indicators: dict[str, Any]         # Indicadores de la última vela cerrada (iloc[-2])


# ---------------------------------------------------------------------------
# Signal — señal de trading emitida por una estrategia
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Señal de trading emitida por una estrategia."""
    symbol: str
    side: str                          # BUY | SELL | HOLD
    size_frac: float                   # Fracción del balance a operar (0.0–1.0)
    strategy_id: str                   # Nombre identificador de la estrategia
    confidence: float = 1.0            # Nivel de confianza (0.0–1.0)
    regime: str = "LATERAL"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.side == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.side == "SELL"

    @property
    def is_hold(self) -> bool:
        return self.side == "HOLD"

    @classmethod
    def hold(cls, symbol: str, strategy_id: str, regime: str = "LATERAL", reason: str = "") -> "Signal":
        """Crea una señal HOLD (no operar)."""
        return cls(
            symbol=symbol,
            side="HOLD",
            size_frac=0.0,
            strategy_id=strategy_id,
            confidence=0.0,
            regime=regime,
            meta={"reason": reason},
        )


# ---------------------------------------------------------------------------
# BaseStrategy — interfaz abstracta
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Clase base para todas las estrategias del sistema multi-estrategia.

    Subclases deben implementar:
      • strategy_id : str — identificador único
      • eligible_symbols : list[str] — símbolos aplicables (vacío = todos)
      • generate_signal(state) -> Signal
    """

    strategy_id: str = "base"
    eligible_symbols: list[str] = []   # vacío = aplica a todos los símbolos
    weight: float = 1.0                # Peso relativo en el PortfolioManager

    def is_eligible(self, symbol: str) -> bool:
        """Retorna True si esta estrategia aplica al símbolo dado."""
        import re
        if not self.eligible_symbols:
            return True
        sym = symbol.upper().strip()
        for pattern in self.eligible_symbols:
            if pattern.startswith("RE:"):
                if re.fullmatch(pattern[3:], sym):
                    return True
            elif sym == pattern.upper():
                return True
        return False

    def generate_signal(self, state: MarketState) -> Signal:
        """Evalúa el estado del mercado y genera una señal.

        Retorna Signal con side=HOLD si no hay oportunidad.
        Subclases deben implementar este método.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} debe implementar generate_signal()"
        )

    def close(self) -> None:
        """Libera recursos (conexiones HTTP, etc.). Override si es necesario."""
