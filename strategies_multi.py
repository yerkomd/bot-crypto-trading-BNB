"""
strategies_multi.py — Motor de estrategias múltiples para bot_trading_v5.py
============================================================================
Implementa 4 estrategias independientes con interfaz común:

  1. TrendFollowingStrategy  — BTC/ETH en tendencia (EMA + ADX + ML)
  2. MeanReversionStrategy   — Altcoins reversión a la media (BB + RSI)
  3. FundingArbitrageStrategy— Arbitraje de funding rate (solo futuros perps)
  4. VolatilityBreakoutStrategy— Breakout en compresión de volatilidad (ATR + BB)

Cada estrategia expone:
  • ELIGIBLE_SYMBOLS  : lista de symbols para los que aplica (regex o lista exacta)
  • signal(df, symbol, regime, balance) -> StrategySignal

El orquestador MultiStrategyEngine evalúa todas las estrategias elegibles para
un símbolo y aplica una lógica de consenso configurable (ANY | MAJORITY | ALL).

INTEGRACIÓN EN bot_trading_v5.py
---------------------------------
1. Importar al inicio del bot:
       from strategies_multi import MultiStrategyEngine, build_default_engine

2. Antes del ThreadPoolExecutor (justo después de inicializar Risk v3):
       MULTI_ENGINE = build_default_engine()

3. En run_strategy(), reemplazar la llamada a strategy.generate_entry(ctx) por:
       sig = MULTI_ENGINE.evaluate(symbol=symbol, df=df, regime=regime,
                                   balance=balance, ctx=ctx)
       entry_ok = sig.should_enter

   O bien en modo híbrido (ML principal + multi como confirmación):
       sig_v5  = strategy.generate_entry(ctx)       # señal original
       sig_ext = MULTI_ENGINE.evaluate(...)          # multi-estrategia
       entry_ok = sig_v5.should_enter and sig_ext.should_enter

VARIABLES DE ENTORNO NUEVAS
----------------------------
STRATEGY_MODE          = ANY | MAJORITY | ALL    (default: ANY)
TREND_ADX_MIN          = mínimo ADX para trend following (default: 25)
TREND_EMA_FAST         = ventana EMA rápida (default: 50)
TREND_EMA_SLOW         = ventana EMA lenta (default: 200)
MEAN_REV_RSI_OB        = RSI sobrecompra altcoins (default: 70)
MEAN_REV_RSI_OS        = RSI sobreventa altcoins (default: 30)
MEAN_REV_BB_WINDOW     = ventana Bollinger Bands (default: 20)
FUNDING_THRESHOLD      = umbral funding rate para señal (default: 0.0005, ≈0.05%)
FUNDING_SYMBOLS        = lista de símbolos para funding arbitrage (default: BTCUSDT,ETHUSDT)
VOL_BREAKOUT_ATR_MULT  = multiplicador ATR para breakout (default: 1.5)
VOL_BREAKOUT_LOOKBACK  = velas de lookback para mínimo ATR (default: 20)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests
import ta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de entorno
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v else default
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v else default
    except Exception:
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = (os.getenv(name) or default).strip()
    return [s.strip().upper() for s in raw.split(',') if s.strip()]


# ---------------------------------------------------------------------------
# Señal de salida unificada
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    should_enter: bool = False
    strategy_name: str = ""
    position_size_frac: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StrategySignal(enter={self.should_enter}, "
            f"strat={self.strategy_name!r}, "
            f"size={self.position_size_frac}, meta={self.meta})"
        )


# ---------------------------------------------------------------------------
# Clase base
# ---------------------------------------------------------------------------

class BaseStrategy:
    name: str = "base"
    ELIGIBLE_SYMBOLS: list[str] = []   # vacío = todos los símbolos

    def is_eligible(self, symbol: str) -> bool:
        if not self.ELIGIBLE_SYMBOLS:
            return True
        sym = symbol.upper().strip()
        for pattern in self.ELIGIBLE_SYMBOLS:
            if pattern.startswith("RE:"):
                if re.fullmatch(pattern[3:], sym):
                    return True
            elif sym == pattern.upper():
                return True
        return False

    def signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        regime: str,
        balance: float,
    ) -> StrategySignal:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. TREND FOLLOWING — BTC / ETH
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(BaseStrategy):
    """
    Señal LONG cuando el precio está en tendencia alcista confirmada.

    Lógica (vela cerrada, iloc[-2]):
      • price > EMA_SLOW                  → precio sobre tendencia larga
      • EMA_FAST > EMA_SLOW               → tendencia alcista de corto plazo
      • ADX >= ADX_MIN                    → tendencia con fuerza
      • RSI en rango [RSI_MIN, RSI_MAX]   → no sobrecomprado
      • Régimen != BEAR                   → filtro macro

    Ideal para BTC y ETH donde la tendencia es más persistente.
    """
    name = "trend_following"

    def __init__(self) -> None:
        self.ema_fast   = _env_int("TREND_EMA_FAST", 50)
        self.ema_slow   = _env_int("TREND_EMA_SLOW", 200)
        self.adx_min    = _env_float("TREND_ADX_MIN", 25.0)
        self.rsi_min    = _env_float("TREND_RSI_MIN", 35.0)
        self.rsi_max    = _env_float("TREND_RSI_MAX", 65.0)
        self.pos_size   = _env_float("TREND_POSITION_SIZE", 0.08)

        # Sólo BTC y ETH por defecto; personalizable vía env
        raw = _env("TREND_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.ELIGIBLE_SYMBOLS = [s.strip().upper() for s in raw.split(',') if s.strip()]

    def signal(self, df: pd.DataFrame, symbol: str, regime: str, balance: float) -> StrategySignal:
        NO = StrategySignal(should_enter=False, strategy_name=self.name)

        if regime == "BEAR":
            return NO

        try:
            row = df.iloc[-2]
            price     = float(row['close'])
            ema_fast  = float(row.get(f'ema{self.ema_fast}', float('nan')))
            ema_slow  = float(row.get(f'ema{self.ema_slow}', float('nan')))
            adx       = float(row.get('adx', float('nan')))
            rsi       = float(row.get('rsi', row.get('rsi14', float('nan'))))
        except Exception as e:
            logger.debug("[%s][trend] error leyendo indicadores: %s", symbol, e)
            return NO

        # Validar NaN
        if any(v != v for v in [price, ema_fast, ema_slow, adx, rsi]):  # NaN check
            return NO

        ok = (
            price > ema_slow
            and ema_fast > ema_slow
            and adx >= self.adx_min
            and self.rsi_min <= rsi <= self.rsi_max
        )

        return StrategySignal(
            should_enter=ok,
            strategy_name=self.name,
            position_size_frac=self.pos_size if ok else None,
            meta={
                "price": price,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "adx": adx,
                "rsi": rsi,
                "regime": regime,
            },
        )


# ---------------------------------------------------------------------------
# 2. MEAN REVERSION — Altcoins
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Señal LONG cuando el precio está sobrevendido y converge hacia la media.

    Lógica (vela cerrada, iloc[-2]):
      • price < BB_LBAND                  → precio bajo banda inferior
      • RSI <= RSI_OVERSOLD               → momentum sobrevendido
      • RSI cruzando hacia arriba         → confirmación de giro
      • Régimen != BEAR                   → filtro macro

    Ideal para altcoins medianas (más reversión a la media que BTC/ETH).
    """
    name = "mean_reversion"

    def __init__(self) -> None:
        self.bb_window  = _env_int("MEAN_REV_BB_WINDOW", 20)
        self.bb_std     = _env_float("MEAN_REV_BB_STD", 2.0)
        self.rsi_os     = _env_float("MEAN_REV_RSI_OS", 30.0)
        self.rsi_ob     = _env_float("MEAN_REV_RSI_OB", 70.0)
        self.rsi_window = _env_int("MEAN_REV_RSI_WINDOW", 14)
        self.pos_size   = _env_float("MEAN_REV_POSITION_SIZE", 0.04)

        # Excluir BTC/ETH por defecto; usar patrón regex para altcoins
        # RE:^(?!BTC|ETH).+USDT$ → cualquier USDT que NO sea BTC ni ETH
        raw = _env("MEAN_REV_SYMBOLS", "RE:^(?!BTC|ETH).+USDT$")
        self.ELIGIBLE_SYMBOLS = [s.strip().upper() for s in raw.split(',') if s.strip()]

    def signal(self, df: pd.DataFrame, symbol: str, regime: str, balance: float) -> StrategySignal:
        NO = StrategySignal(should_enter=False, strategy_name=self.name)

        if regime == "BEAR":
            return NO

        try:
            # Calcular BB y RSI si no vienen del df principal
            close = df['close']
            bb = ta.volatility.BollingerBands(close=close, window=self.bb_window, window_dev=self.bb_std)
            bb_lband = bb.bollinger_lband()

            rsi_ind = ta.momentum.RSIIndicator(close=close, window=self.rsi_window)
            rsi_series = rsi_ind.rsi()

            row_idx = -2
            price      = float(close.iloc[row_idx])
            lband      = float(bb_lband.iloc[row_idx])
            rsi_now    = float(rsi_series.iloc[row_idx])
            rsi_prev   = float(rsi_series.iloc[row_idx - 1])
        except Exception as e:
            logger.debug("[%s][mean_rev] error calculando: %s", symbol, e)
            return NO

        if any(v != v for v in [price, lband, rsi_now, rsi_prev]):
            return NO

        # RSI cruzando desde zona sobreventa: estaba por debajo del umbral y ahora sube
        rsi_crossing_up = rsi_prev <= self.rsi_os and rsi_now > rsi_prev
        ok = (
            price < lband
            and rsi_now <= self.rsi_os
            and rsi_crossing_up
        )

        return StrategySignal(
            should_enter=ok,
            strategy_name=self.name,
            position_size_frac=self.pos_size if ok else None,
            meta={
                "price": price,
                "bb_lband": lband,
                "rsi": rsi_now,
                "rsi_prev": rsi_prev,
                "regime": regime,
            },
        )


# ---------------------------------------------------------------------------
# 3. FUNDING ARBITRAGE — Perps con funding rate negativo/positivo extremo
# ---------------------------------------------------------------------------

class FundingArbitrageStrategy(BaseStrategy):
    """
    Filtro macro basado en funding rate de futuros perpetuos (spot-safe).

    Para un bot spot esta estrategia actúa como guardia de sentimiento:
      • Funding >  THRESHOLD → mercado sobreexpuesto en longs → bloquear entrada
      • Funding <= THRESHOLD → sentimiento neutro o bajista   → permitir entrada
      • Sin datos / API down → fail-open (no bloquear trading)
      • Régimen BEAR         → bloquear entrada (consistente con otras estrategias)

    No genera señales de entrada propias — en modo MAJORITY/ALL actúa como veto.
    Fuente: Binance Futures API (GET /fapi/v1/fundingRate). Caché de 1h.
    """
    name = "funding_arb"
    _CACHE: dict[str, dict] = {}              # {symbol: {rate: float, ts: float}}
    _CACHE_LOCK: threading.Lock = threading.Lock()  # protege _CACHE en multi-hilo
    _CACHE_TTL_S: float = 3600.0              # refrescar cada hora (funding cambia cada 8h)

    def __init__(self) -> None:
        self.threshold  = _env_float("FUNDING_THRESHOLD", 0.0005)
        self._session   = requests.Session()

        raw = _env("FUNDING_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.ELIGIBLE_SYMBOLS = [s.strip().upper() for s in raw.split(',') if s.strip()]

    def close(self) -> None:
        """Cierra la sesión HTTP. Llamar en shutdown del bot."""
        self._session.close()

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Retorna el funding rate actual. Usa caché thread-safe de 1h."""
        now = time.time()
        with FundingArbitrageStrategy._CACHE_LOCK:
            cached = FundingArbitrageStrategy._CACHE.get(symbol)
            if cached and (now - cached.get('ts', 0)) < self._CACHE_TTL_S:
                return float(cached['rate'])

        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        try:
            resp = self._session.get(
                url,
                params={"symbol": symbol, "limit": 1},
                timeout=8.0,
                headers={"Accept": "application/json"},
            )
            data = resp.json()
            if data and isinstance(data, list):
                rate = float(data[-1].get('fundingRate', 0.0))
                with FundingArbitrageStrategy._CACHE_LOCK:
                    FundingArbitrageStrategy._CACHE[symbol] = {'rate': rate, 'ts': now}
                logger.debug("[%s] funding rate: %.6f", symbol, rate)
                return rate
        except Exception as e:
            logger.debug("[%s][funding] fetch falló: %s", symbol, e)
        return None

    def signal(self, df: pd.DataFrame, symbol: str, regime: str, balance: float) -> StrategySignal:
        PASS = StrategySignal(should_enter=True, strategy_name=self.name)
        NO   = StrategySignal(should_enter=False, strategy_name=self.name)

        if regime == "BEAR":
            return NO

        rate = self._fetch_funding_rate(symbol)
        if rate is None:
            return PASS  # fail-open: API caída no debe bloquear trading spot

        # Bloquear solo si el mercado futures está sobreexpuesto en longs
        ok = rate <= self.threshold

        return StrategySignal(
            should_enter=ok,
            strategy_name=self.name,
            position_size_frac=None,  # filtro macro: no sugiere tamaño de posición
            meta={
                "funding_rate": rate,
                "threshold": self.threshold,
                "blocked": not ok,
                "regime": regime,
            },
        )


# ---------------------------------------------------------------------------
# 4. VOLATILITY BREAKOUT — Compresión → Expansión
# ---------------------------------------------------------------------------

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Señal LONG cuando el precio rompe al alza tras un período de baja volatilidad.

    Lógica (vela cerrada):
      • ATR actual en mínimo de últimas N velas        → compresión detectada
      • Precio cierra por encima de máximo reciente    → breakout confirmado
      • Volumen superior a media                        → breakout con convicción
      • BB Width en expansión vs período anterior       → volatilidad expandiéndose

    Funciona para cualquier símbolo pero especialmente efectivo en BTC/ETH y
    altcoins de alta liquidez en momentos de consolidación.
    """
    name = "vol_breakout"

    def __init__(self) -> None:
        self.atr_mult       = _env_float("VOL_BREAKOUT_ATR_MULT", 1.5)
        self.lookback       = _env_int("VOL_BREAKOUT_LOOKBACK", 20)
        self.vol_mult       = _env_float("VOL_BREAKOUT_VOL_MULT", 1.2)   # volumen > N x media
        self.bb_window      = _env_int("VOL_BREAKOUT_BB_WINDOW", 20)
        self.bb_expansion_lag = _env_int("VOL_BREAKOUT_BB_LAG", 3)       # velas atrás para comparar BB width
        self.pos_size       = _env_float("VOL_BREAKOUT_POSITION_SIZE", 0.06)

        raw = _env("VOL_BREAKOUT_SYMBOLS", "")  # vacío = todos los símbolos
        self.ELIGIBLE_SYMBOLS = [s.strip().upper() for s in raw.split(',') if s.strip()]

    def signal(self, df: pd.DataFrame, symbol: str, regime: str, balance: float) -> StrategySignal:
        NO = StrategySignal(should_enter=False, strategy_name=self.name)

        if regime == "BEAR":
            return NO

        try:
            # ATR y breakout de precio
            atr_series  = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14
            ).average_true_range()

            close     = df['close']
            high      = df['high']
            volume    = df['volume']

            idx = -2   # vela cerrada
            atr_now   = float(atr_series.iloc[idx])
            atr_min   = float(atr_series.iloc[idx - self.lookback: idx].min())
            price_now = float(close.iloc[idx])

            # Máximo de las últimas N velas (excluyendo la actual)
            recent_high = float(high.iloc[idx - self.lookback: idx].max())

            # Volumen
            vol_now  = float(volume.iloc[idx])
            vol_mean = float(volume.iloc[idx - self.lookback: idx].mean())

            # BB Width
            bb = ta.volatility.BollingerBands(close=close, window=self.bb_window, window_dev=2)
            bb_width_now  = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[idx])
            bb_width_prev = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[idx - self.bb_expansion_lag])

        except Exception as e:
            logger.debug("[%s][vol_breakout] error: %s", symbol, e)
            return NO

        if any(v != v for v in [atr_now, atr_min, price_now, recent_high, vol_now, vol_mean]):
            return NO

        compresion   = atr_now <= atr_min * self.atr_mult   # ATR comprimido
        breakout_up  = price_now > recent_high               # rompe máximo reciente
        vol_confirm  = vol_now >= vol_mean * self.vol_mult   # volumen superior
        bb_expanding = bb_width_now > bb_width_prev          # BB expandiéndose

        ok = compresion and breakout_up and vol_confirm and bb_expanding

        return StrategySignal(
            should_enter=ok,
            strategy_name=self.name,
            position_size_frac=self.pos_size if ok else None,
            meta={
                "atr_now": atr_now,
                "atr_min": atr_min,
                "price": price_now,
                "recent_high": recent_high,
                "vol_now": vol_now,
                "vol_mean": vol_mean,
                "bb_expanding": bb_expanding,
                "regime": regime,
            },
        )


# ---------------------------------------------------------------------------
# Orquestador: MultiStrategyEngine
# ---------------------------------------------------------------------------

class MultiStrategyEngine:
    """
    Evalúa múltiples estrategias para un símbolo y combina sus señales.

    Modos de consenso (STRATEGY_MODE env var):
      ANY      → basta con que UNA estrategia diga "entrar"  (más señales)
      MAJORITY → la mayoría de estrategias elegibles deben coincidir
      ALL      → TODAS las estrategias elegibles deben coincidir (más estricto)

    Recomendación por contexto:
      • Capital pequeño / alta actividad   → ANY
      • Capital medio / balanceado         → MAJORITY  ← default sugerido
      • Capital grande / preservar capital → ALL
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        mode: str = "ANY",
    ) -> None:
        self.strategies = strategies
        self.mode = mode.upper().strip()
        if self.mode not in ("ANY", "MAJORITY", "ALL"):
            logger.warning("STRATEGY_MODE='%s' inválido, usando ANY", self.mode)
            self.mode = "ANY"
        logger.info(
            "MultiStrategyEngine init: mode=%s strategies=[%s]",
            self.mode,
            ", ".join(s.name for s in self.strategies),
        )

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str,
        balance: float,
        ctx=None,  # StrategyContext opcional para compatibilidad con bot_trading_v5
    ) -> StrategySignal:
        """
        Evalúa todas las estrategias elegibles y retorna señal consolidada.
        """
        eligible = [s for s in self.strategies if s.is_eligible(symbol)]
        if not eligible:
            return StrategySignal(should_enter=False, strategy_name="none_eligible", meta={"symbol": symbol})

        signals: list[StrategySignal] = []
        for strat in eligible:
            try:
                sig = strat.signal(df=df, symbol=symbol, regime=regime, balance=balance)
                signals.append(sig)
                logger.debug(
                    "[%s][%s] → enter=%s meta=%s",
                    symbol, strat.name, sig.should_enter, sig.meta,
                )
            except Exception as e:
                logger.warning("[%s][%s] excepción en signal(): %s", symbol, strat.name, e)
                signals.append(StrategySignal(should_enter=False, strategy_name=strat.name))

        positive = [s for s in signals if s.should_enter]
        n_eligible = len(signals)
        n_positive = len(positive)

        if self.mode == "ANY":
            ok = n_positive >= 1
        elif self.mode == "MAJORITY":
            ok = n_positive > n_eligible / 2
        else:  # ALL
            ok = n_positive == n_eligible and n_eligible > 0

        # Tamaño de posición: mínimo de los sugeridos (conservador, evita sobreexposición)
        sizes = [s.position_size_frac for s in positive if s.position_size_frac is not None]
        pos_size = min(sizes) if sizes else None

        # Nombre de estrategias que dispararon
        triggered = [s.strategy_name for s in positive]

        logger.debug(
            "[%s] MultiEngine: mode=%s eligible=%d positive=%d enter=%s triggered=%s",
            symbol, self.mode, n_eligible, n_positive, ok, triggered,
        )

        return StrategySignal(
            should_enter=ok,
            strategy_name="+".join(triggered) if triggered else "none",
            position_size_frac=pos_size,
            meta={
                "mode": self.mode,
                "n_eligible": n_eligible,
                "n_positive": n_positive,
                "triggered": triggered,
                "signals": [
                    {"name": s.strategy_name, "enter": s.should_enter, "meta": s.meta}
                    for s in signals
                ],
            },
        )


# ---------------------------------------------------------------------------
# Factory: construye el engine con defaults desde env vars
# ---------------------------------------------------------------------------

def build_default_engine() -> MultiStrategyEngine:
    """
    Construye MultiStrategyEngine con las 4 estrategias.
    Llama esta función en main() de bot_trading_v5.py.

    Ejemplo:
        MULTI_ENGINE = build_default_engine()
    """
    mode = _env("STRATEGY_MODE", "MAJORITY")
    strategies: list[BaseStrategy] = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        FundingArbitrageStrategy(),
        VolatilityBreakoutStrategy(),
    ]
    return MultiStrategyEngine(strategies=strategies, mode=mode)


# ---------------------------------------------------------------------------
# Uso como script de prueba (sin conexión a Binance)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)

    # Generar df de prueba con datos sintéticos
    n = 300
    np.random.seed(42)
    price = 50_000 + np.cumsum(np.random.randn(n) * 200)
    df_test = pd.DataFrame({
        "open":   price * (1 - 0.002),
        "high":   price * 1.005,
        "low":    price * 0.995,
        "close":  price,
        "volume": np.random.uniform(100, 500, n),
    })

    # Añadir indicadores básicos para TrendFollowing
    df_test['ema50']  = ta.trend.EMAIndicator(df_test['close'], window=50).ema_indicator()
    df_test['ema200'] = ta.trend.EMAIndicator(df_test['close'], window=200).ema_indicator()
    df_test['adx']    = ta.trend.ADXIndicator(
        high=df_test['high'], low=df_test['low'], close=df_test['close'], window=14
    ).adx()
    df_test['rsi']    = ta.momentum.RSIIndicator(df_test['close'], window=14).rsi()
    df_test['rsi14']  = df_test['rsi']

    engine = build_default_engine()

    for sym in ["BTCUSDT", "SOLUSDT", "ADAUSDT"]:
        result = engine.evaluate(symbol=sym, df=df_test, regime="BULL", balance=1000.0)
        print(f"\n[{sym}] → enter={result.should_enter} | strats={result.strategy_name}")
        print(f"  pos_size={result.position_size_frac}")
        for s in result.meta.get("signals", []):
            print(f"    {s['name']:25s} enter={s['enter']}")
