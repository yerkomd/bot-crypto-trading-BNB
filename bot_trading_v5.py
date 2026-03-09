from binance.client import Client
import time
import pandas as pd
import os
import sys
import logging
import logging.handlers
import ta
import joblib
from decimal import Decimal, ROUND_DOWN
import requests # Nuevo import
from dotenv import load_dotenv # Nuevo import para cargar variables de entorno
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path

from db.connection import init_db_from_env
from db.schema import ensure_schema
from repositories.open_positions_repo import OpenPositionsRepository
from repositories.trade_history_repo import TradeHistoryRepository
from repositories.equity_repo import EquitySnapshotsRepository
from services.market_klines_service import read_klines_df

# Risk Layer v2 (institucional) - modular, no cambia el flujo de estrategia
from risk_layer_v2 import (
    GlobalRiskController,
    RiskEventLogger,
    SystemHealthMonitor,
    reconcile_positions_with_exchange,
    reconcile_worker,
    risk_metrics_worker,
    start_health_server,
)

from risk_layer_v3 import (
    EquityRegimeFilter,
    IntradayVaRMonitor,
    PortfolioCorrelationRisk,
    SlippageMonitor,
    VolatilityPositionSizer,
)

from backtesting.bt_types import StrategyContext

# Cargar variables de entorno desde un archivo .env (no toca red; seguro para imports/tests)
load_dotenv()

STOP_EVENT = threading.Event()

# Fallback robusto para excepciones de la lib
try:
    from binance.exceptions import BinanceAPIException
except Exception:
    class BinanceAPIException(Exception):
        pass

# Configuración de logging: salida a stdout (para docker logs) + archivo rotativo
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Crear directorio de logs
os.makedirs('./logs', exist_ok=True)


# Configurar el logger root
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Para docker logs
        logging.handlers.RotatingFileHandler(
            './logs/bot_trading.log', 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
    ]
)

logger = logging.getLogger(__name__)
logger.info("Bot Trading v2 iniciado. LOG_LEVEL=%s", LOG_LEVEL)

# Throttled log helper (avoid spamming on every loop)
_THROTTLED_LOG_TS: dict[str, float] = {}


def _log_throttled(key: str, level: int, msg: str, *args, every_s: float = 600.0) -> None:
    now = time.time()
    last = _THROTTLED_LOG_TS.get(key)
    if last is not None and (now - last) < every_s:
        return
    _THROTTLED_LOG_TS[key] = now
    logger.log(level, msg, *args)

# --- Persistencia (PostgreSQL) ---
DB = None
OPEN_POS_REPO: OpenPositionsRepository | None = None
TRADE_REPO: TradeHistoryRepository | None = None
EQUITY_REPO: EquitySnapshotsRepository | None = None

RISK_EVENT_LOGGER: RiskEventLogger | None = None
GLOBAL_RISK_CONTROLLER: GlobalRiskController | None = None
SYSTEM_HEALTH_MONITOR: SystemHealthMonitor | None = None

V3_POSITION_SIZER: VolatilityPositionSizer | None = None
V3_CORRELATION_RISK: PortfolioCorrelationRisk | None = None
V3_VAR_MONITOR: IntradayVaRMonitor | None = None
V3_SLIPPAGE_MONITOR: SlippageMonitor | None = None
V3_EQUITY_REGIME_FILTER: EquityRegimeFilter | None = None

# Motor multi-estrategia legacy (inicializado en main())
from strategies_multi import MultiStrategyEngine, build_default_engine
MULTI_ENGINE: MultiStrategyEngine | None = None

# Arquitectura multi-estrategia v2 (StrategyEngine + PortfolioManager)
from strategy_engine import StrategyEngine
from portfolio_manager import PortfolioManager, PortfolioOrder
STRATEGY_ENGINE: StrategyEngine | None = None
PORTFOLIO_MANAGER: PortfolioManager | None = None

POSITIONS_CACHE_BY_SYMBOL: dict[str, list[dict]] = {}

def _require_repos():
    if OPEN_POS_REPO is None or TRADE_REPO is None or EQUITY_REPO is None:
        raise RuntimeError("Repositorios DB no inicializados. Verifica USE_DATABASE y la conexión a PostgreSQL.")


# Configuración de Binance
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_SECRET')
# --- Configuración de Telegram (NUEVO) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') # Carga desde .env
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')     # Carga desde .env

def _env_float(name, default):
    try:
        v = os.getenv(name)
        return float(v) if v is not None and v != '' else float(default)
    except Exception:
        logger.warning(f"Variable de entorno {name} inválida, usando {default}")
        return float(default)

# Parámetros con valores por defecto
SYMBOLS = [s.strip().upper() for s in os.getenv('SYMBOLS', 'BTCUSDT').split(',') if s.strip()]
rsi_threshold = _env_float('RSI_THRESHOLD', 30.0)
take_profit_pct = _env_float('TAKE_PROFIT_PCT', 2.0)
stop_loss_pct = _env_float('STOP_LOSS_PCT', 1.0)
trailing_take_profit_pct = _env_float('TRAILING_TAKE_PROFIT_PCT', 0.5)
trailing_stop_pct = _env_float('TRAILING_STOP_PCT', 0.5)
position_size = _env_float('POSITION_SIZE', 0.01)
timeframe = os.getenv('TIMEFRAME', '1h')
step_size = float(os.getenv('STEP_SIZE', 0.00001))
min_notional = float(os.getenv('MIN_NOTIONAL', 10))

# --- ML entry filter (v5) ---
# Modelo offline (sin llamadas de red). Si no carga o faltan features => NO entrar.
ML_MODEL_PATH = str(os.getenv('ML_MODEL_PATH', 'artifacts/model_momentum_v4_phase2.joblib')).strip()
try:
    ML_PROB_THRESHOLD = float(os.getenv('ML_PROB_THRESHOLD', '0.60'))
except Exception:
    ML_PROB_THRESHOLD = 0.60

_ML_BUNDLE: dict | None = None
_ML_LOAD_ERROR: str | None = None
_ML_LOCK = threading.Lock()

# Definición anticipada: ATR_WINDOW es usado por _ensure_ml_features.
# Python resuelve globales en tiempo de llamada (no de definición), pero declararlo
# aquí hace la dependencia explícita y previene NameError si el orden del módulo cambia.
ATR_WINDOW = int(os.getenv('ATR_WINDOW', '14'))

# Caché TTL para btc_dominance (CoinGecko): evita llamadas repetidas en cada ciclo.
_BTC_DOMINANCE_CACHE: dict = {'value': None, 'ts': 0.0}
_BTC_DOMINANCE_TTL_S: float = 3600.0  # refrescar cada hora


def _ensure_ml_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Asegura columnas necesarias para el modelo ML (solo desde OHLCV/indicadores locales).

    NOTA:
    - No hace llamadas de red.
    - Si alguna feature no es computable con la data disponible, NO se crea.
      (Así, get_ml_probability puede fail-closed como requiere el spec.)
    """
    if df is None or df.empty or not features:
        return df

    out = df

    # Base indicators we can compute locally
    try:
        if 'ema200' not in out.columns:
            out['ema200'] = ta.trend.EMAIndicator(out['close'], window=200).ema_indicator()
        if 'ema50' not in out.columns:
            out['ema50'] = ta.trend.EMAIndicator(out['close'], window=50).ema_indicator()
        if 'atr' not in out.columns:
            atr_ind = ta.volatility.AverageTrueRange(high=out['high'], low=out['low'], close=out['close'], window=ATR_WINDOW)
            out['atr'] = atr_ind.average_true_range()
        if 'adx' not in out.columns:
            try:
                adx_ind = ta.trend.ADXIndicator(high=out['high'], low=out['low'], close=out['close'], window=14)
                out['adx'] = adx_ind.adx()
            except Exception:
                out['adx'] = pd.NA
    except Exception:
        return out

    fset = set(str(x) for x in features)

    # Common engineered features (v1/v2/v3 models)
    if 'dist_price_ema200' in fset and 'dist_price_ema200' not in out.columns:
        try:
            out['dist_price_ema200'] = (out['close'] - out['ema200']) / out['ema200']
        except Exception:
            pass

    if 'ema50_slope' in fset and 'ema50_slope' not in out.columns:
        try:
            w = 5
            out['ema50_slope'] = (out['ema50'] - out['ema50'].shift(w)) / out['ema50'].shift(w)
        except Exception:
            pass

    if 'adx14' in fset and 'adx14' not in out.columns:
        try:
            out['adx14'] = out['adx']
        except Exception:
            pass

    if 'atr_norm' in fset and 'atr_norm' not in out.columns:
        try:
            out['atr_norm'] = out['atr'] / out['close']
        except Exception:
            pass

    if 'vol_rel_20' in fset and 'vol_rel_20' not in out.columns:
        try:
            vol_ma = out['volume'].rolling(window=20).mean()
            out['vol_rel_20'] = out['volume'] / vol_ma
        except Exception:
            pass

    if 'rsi14' in fset and 'rsi14' not in out.columns:
        try:
            if 'rsi' in out.columns:
                out['rsi14'] = out['rsi']
            else:
                out['rsi14'] = ta.momentum.RSIIndicator(out['close'], window=14).rsi()
        except Exception:
            pass

    if 'macd_signal' in fset and 'macd_signal' not in out.columns:
        try:
            macd_indicator = ta.trend.MACD(out['close'])
            out['macd_signal'] = macd_indicator.macd_signal()
        except Exception:
            pass

    if 'bb_width_norm' in fset and 'bb_width_norm' not in out.columns:
        try:
            bb = ta.volatility.BollingerBands(close=out['close'], window=20, window_dev=2)
            width = bb.bollinger_hband() - bb.bollinger_lband()
            out['bb_width_norm'] = width / out['close']
        except Exception:
            pass

    # Lags (only if needed)
    for base, lag_name in (
        ('atr_norm', 'atr_norm_lag1'),
        ('vol_rel_20', 'vol_rel_20_lag1'),
        ('rsi14', 'rsi14_lag1'),
    ):
        if lag_name in fset and lag_name not in out.columns and base in out.columns:
            try:
                out[lag_name] = out[base].shift(1)
            except Exception:
                pass

    # Simple interactions (only if needed)
    if 'atr_vol_interaction' in fset and 'atr_vol_interaction' not in out.columns:
        if 'atr_norm' in out.columns and 'vol_rel_20' in out.columns:
            try:
                out['atr_vol_interaction'] = out['atr_norm'] * out['vol_rel_20']
            except Exception:
                pass

    if 'rsi_macd_interaction' in fset and 'rsi_macd_interaction' not in out.columns:
        if 'rsi14' in out.columns and 'macd_signal' in out.columns:
            try:
                out['rsi_macd_interaction'] = out['rsi14'] * out['macd_signal']
            except Exception:
                pass

    # volatility_regime: 1.0 si ATR actual > media ATR de 30 velas, 0.0 si no.
    # Captura si el mercado está en régimen de alta volatilidad.
    if 'volatility_regime' in fset and 'volatility_regime' not in out.columns:
        try:
            if 'atr' in out.columns:
                atr_mean = out['atr'].rolling(window=30, min_periods=10).mean()
                out['volatility_regime'] = (out['atr'] > atr_mean).astype(float)
        except Exception:
            pass

    # market_breadth: proxy single-asset = fracción de últimas 20 velas donde close > EMA200.
    # Aproxima la "amplitud" del mercado usando la tendencia del propio símbolo.
    # eth_correlation_30 y btc_dominance se computan en _add_cross_asset_features (requieren
    # datos externos/DB) y llegan ya incluidos en el df antes de llamar a esta función.
    if 'market_breadth' in fset and 'market_breadth' not in out.columns:
        try:
            if 'ema200' in out.columns:
                out['market_breadth'] = (
                    out['close'].gt(out['ema200'])
                    .rolling(window=20, min_periods=5)
                    .mean()
                )
        except Exception:
            pass

    return out


def _get_ml_bundle() -> dict | None:
    """Carga lazy el artefacto joblib con el modelo y metadata.

    Formatos soportados:
    - dict con keys: model, features, operating_threshold, signals_enabled, operating_mode
    - modelo sklearn/pipeline directamente (features via feature_names_in_ o fallback)
    """
    global _ML_BUNDLE, _ML_LOAD_ERROR
    if _ML_BUNDLE is not None or _ML_LOAD_ERROR is not None:
        return _ML_BUNDLE

    with _ML_LOCK:
        if _ML_BUNDLE is not None or _ML_LOAD_ERROR is not None:
            return _ML_BUNDLE

        path = Path(ML_MODEL_PATH)
        try:
            obj = joblib.load(path)
        except Exception as e:
            _ML_LOAD_ERROR = f"{type(e).__name__}: {e}"
            logger.error("ML model no pudo cargarse (%s). Entradas bloqueadas.", _ML_LOAD_ERROR)
            return None

        bundle: dict
        if isinstance(obj, dict):
            bundle = dict(obj)
            # Support multiple artifact schemas
            if bundle.get('model') is None and isinstance(bundle.get('production_model'), dict):
                bundle['model'] = bundle['production_model'].get('model')
        else:
            bundle = {'model': obj}

        model = bundle.get('model')
        features = bundle.get('features')
        if features is None:
            try:
                fin = getattr(model, 'feature_names_in_', None)
                if fin is not None:
                    features = list(fin)
            except Exception:
                features = None
        if features is None:
            # Fallback: features del modelo v4_phase2 ordenadas por importancia descendente.
            # IMPORTANTE: el artefacto .joblib debería siempre incluir 'features' en el bundle
            # para evitar este fallback. Si no lo incluye, se asume este conjunto entrenado.
            features = [
                'atr_vol_interaction', 'atr_norm', 'atr_norm_lag1', 'bb_width_norm',
                'macd_signal', 'dist_price_ema200', 'vol_rel_20', 'btc_dominance',
                'ema50_slope', 'rsi14', 'rsi_macd_interaction', 'eth_correlation_30',
                'vol_rel_20_lag1', 'adx14', 'rsi14_lag1', 'market_breadth',
                'volatility_regime',
            ]

        bundle.setdefault('signals_enabled', True)
        bundle.setdefault('operating_mode', 'active')
        bundle.setdefault('operating_threshold', ML_PROB_THRESHOLD)
        bundle['features'] = list(features)

        _ML_BUNDLE = bundle
        logger.info(
            "ML model cargado: path=%s features=%s threshold=%.3f mode=%s enabled=%s",
            str(path),
            ",".join(bundle['features']),
            float(bundle.get('operating_threshold') or ML_PROB_THRESHOLD),
            str(bundle.get('operating_mode')),
            bool(bundle.get('signals_enabled')),
        )
        return _ML_BUNDLE


def get_ml_probability(df_row) -> float | None:
    """Retorna probabilidad clase positiva (1) o None si no se puede evaluar.

    Validaciones:
    - modelo debe estar disponible
    - features deben existir y no ser NaN
    - modelo debe exponer predict_proba
    """
    bundle = _get_ml_bundle()
    if not bundle:
        return None

    if not bool(bundle.get('signals_enabled', True)):
        return None
    if str(bundle.get('operating_mode', 'active')).strip().lower() != 'active':
        return None

    model = bundle.get('model')
    if model is None or not hasattr(model, 'predict_proba'):
        _log_throttled('ml_no_predict_proba', logging.ERROR, "ML model sin predict_proba; entradas bloqueadas", every_s=600)
        return None

    features = list(bundle.get('features') or [])
    if not features:
        return None

    values: dict[str, float] = {}
    missing = []
    for f in features:
        try:
            v = df_row.get(f) if hasattr(df_row, 'get') else None
        except Exception:
            v = None
        if v is None or (hasattr(pd, 'isna') and pd.isna(v)):
            missing.append(str(f))
            continue
        try:
            fv = float(v)
        except Exception:
            missing.append(str(f))
            continue
        # Rechazar NaN e inf/-inf: fv != fv es True solo para NaN (IEEE 754)
        if fv != fv or fv == float('inf') or fv == float('-inf'):
            missing.append(str(f))
            continue
        values[str(f)] = fv

    if missing:
        # Clave estática para evitar crecimiento ilimitado de _THROTTLED_LOG_TS
        # (clave dinámica con missing[:4] generaría una entrada por cada combinación distinta).
        _log_throttled(
            'ml_missing_features',
            logging.WARNING,
            "ML: faltan features (%s) -> no entrar",
            ",".join(missing),
            every_s=120,
        )
        return None

    row_df = pd.DataFrame([values], columns=features)
    try:
        proba = model.predict_proba(row_df)
        # Esperado: [[p0, p1]] para binario
        if hasattr(proba, '__len__') and len(proba) > 0 and len(proba[0]) >= 2:
            p = float(proba[0][1])
        else:
            p = float(proba[0][0])
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p
    except Exception as e:
        _log_throttled('ml_predict_error', logging.WARNING, "ML predict_proba falló: %s", str(e), every_s=120)
        return None


def _fetch_btc_dominance() -> float | None:
    """Devuelve la dominancia de BTC (%) desde CoinGecko, con caché de TTL=1h.

    Si la API falla, retorna el último valor cacheado (stale) o None.
    En caso de None, get_ml_probability detecta la feature faltante y retorna None (fail-closed).
    """
    global _BTC_DOMINANCE_CACHE
    now = time.time()
    cached_val = _BTC_DOMINANCE_CACHE.get('value')
    cached_ts = float(_BTC_DOMINANCE_CACHE.get('ts') or 0.0)
    if cached_val is not None and (now - cached_ts) < _BTC_DOMINANCE_TTL_S:
        return float(cached_val)
    try:
        resp = _HTTP_SESSION.get(
            'https://api.coingecko.com/api/v3/global',
            timeout=8.0,
            headers={'Accept': 'application/json'},
        )
        data = resp.json()
        btc_dom = float(data['data']['market_cap_percentage']['btc'])
        _BTC_DOMINANCE_CACHE = {'value': btc_dom, 'ts': now}
        logger.debug("btc_dominance actualizado: %.2f%%", btc_dom)
        return btc_dom
    except Exception as e:
        _log_throttled(
            'btc_dom_fail', logging.WARNING,
            'btc_dominance fetch falló (stale=%s): %s', str(cached_val), str(e),
            every_s=300,
        )
        return float(cached_val) if cached_val is not None else None


def _add_cross_asset_features(
    df: pd.DataFrame, *, symbol: str, interval: str, features: list[str]
) -> pd.DataFrame:
    """Añade features que requieren datos externos o cross-asset (no computables solo desde OHLCV).

    Features computadas:
    - eth_correlation_30: correlación rolling 30 entre retornos del símbolo y ETH (o BTC si es ETH).
      Requiere que ETHUSDT/BTCUSDT esté sincronizado en trading.market_klines por market_data_process.
    - btc_dominance: dominancia BTC en % (CoinGecko, cacheada 1h).

    Fail-open por feature: si falla una, se omite sin propagar excepción.
    get_ml_probability detectará la feature faltante y retornará None (fail-closed).
    """
    if df is None or df.empty:
        return df

    fset = set(str(x) for x in (features or []))
    out = df

    # 1. eth_correlation_30
    if 'eth_correlation_30' in fset and 'eth_correlation_30' not in out.columns:
        try:
            if DB is not None:
                sym_u = str(symbol).upper().strip()
                cross_sym = 'BTCUSDT' if sym_u == 'ETHUSDT' else 'ETHUSDT'
                n = len(out)
                cross_df = read_klines_df(db=DB, symbol=cross_sym, interval=interval, limit=n + 5)
                if cross_df is not None and not cross_df.empty:
                    target_ret = out['close'].pct_change().reset_index(drop=True)
                    cross_ret = cross_df['close'].pct_change().iloc[-n:].reset_index(drop=True)
                    min_len = min(len(target_ret), len(cross_ret))
                    corr = (
                        target_ret.iloc[-min_len:]
                        .reset_index(drop=True)
                        .rolling(window=30, min_periods=15)
                        .corr(cross_ret.iloc[-min_len:].reset_index(drop=True))
                    )
                    out = out.copy()
                    col = pd.Series([float('nan')] * len(out), index=out.index, dtype=float)
                    col.iloc[-min_len:] = corr.values
                    out['eth_correlation_30'] = col
        except Exception as e:
            _log_throttled(
                'eth_corr_fail', logging.WARNING,
                '[%s] eth_correlation_30 falló: %s', symbol, str(e),
                every_s=300,
            )

    # 2. btc_dominance (valor escalar; se rellena toda la columna con el mismo valor)
    if 'btc_dominance' in fset and 'btc_dominance' not in out.columns:
        dom = _fetch_btc_dominance()
        if dom is not None:
            if out is df:
                out = out.copy()
            out['btc_dominance'] = float(dom)

    return out


# Variables para control de frecuencia de compras
cooldown_seconds = int(os.getenv('BUY_COOLDOWN_SECONDS', '5400'))  # 90 min por defecto
poll_interval = int(os.getenv('POLL_INTERVAL_SECONDS', '30'))      # frecuencia de chequeo

# Snapshots de equity (time-series en PostgreSQL)
EQUITY_SNAPSHOT_INTERVAL = int(os.getenv('EQUITY_SNAPSHOT_INTERVAL', '300'))

# --- Estrategia / Riesgo (nuevo, sin cambiar arquitectura) ---
ATR_WINDOW = int(os.getenv('ATR_WINDOW', '14'))

RSI_CONFIRM_MIN = float(os.getenv('RSI_CONFIRM_MIN', '35'))
RSI_CONFIRM_MAX = float(os.getenv('RSI_CONFIRM_MAX', '55'))

MAX_SYMBOL_DRAWDOWN_FRAC = float(os.getenv('MAX_SYMBOL_DRAWDOWN_FRAC', '0.06'))  # 6%
MAX_CAPITAL_COMMITTED_FRAC = float(os.getenv('MAX_CAPITAL_COMMITTED_FRAC', '0.15'))  # 15% del equity total
DAILY_LOSS_LIMIT_FRAC = float(os.getenv('DAILY_LOSS_LIMIT_FRAC', '0.03'))  # 3% del equity total (día)
SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS = int(os.getenv('SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS', str(24 * 3600)))

# DCA: por defecto desactivado (evita martingala). Se puede activar explícitamente.
ENABLE_DCA = str(os.getenv('ENABLE_DCA', '0')).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

# Multiplicadores ATR por régimen: TP/SL y trailing-stop (solo para nuevas posiciones)
# Configurables via env vars: BULL_TP_MULT, BULL_SL_MULT, BULL_TRAIL_MULT (y equivalentes LATERAL_*)
ATR_MULTIPLIERS = {
    "BULL": {
        "tp":          float(os.getenv("BULL_TP_MULT",    "4.0")),
        "sl":          float(os.getenv("BULL_SL_MULT",    "1.5")),
        "trailing_sl": float(os.getenv("BULL_TRAIL_MULT", "2.0")),
    },
    "LATERAL": {
        "tp":          float(os.getenv("LATERAL_TP_MULT",    "2.0")),
        "sl":          float(os.getenv("LATERAL_SL_MULT",    "1.2")),
        "trailing_sl": float(os.getenv("LATERAL_TRAIL_MULT", "1.0")),
    },
    "BEAR": None,
}

# ADX mínimo para considerar régimen como tendencia fuerte (BULL/BEAR).
# Umbral elevado (35) para reducir entradas en rallies de mercado bajista.
ADX_REGIME_MIN = float(os.getenv("ADX_REGIME_MIN", "35.0"))

# Filtro pendiente EMA200: True = solo entrar cuando EMA200 tiene tendencia alcista.
EMA200_SLOPE_FILTER = str(os.getenv("EMA200_SLOPE_FILTER", "true")).lower() == "true"
EMA200_SLOPE_BARS   = int(os.getenv("EMA200_SLOPE_BARS", "10"))

# Si hay balance bloqueado en órdenes abiertas, un STOP LOSS podría no poder vender.
# Si activas esta opción, el bot intentará cancelar órdenes abiertas del símbolo para liberar balance.
# 0/false = desactivado (default), 1/true = activado
cancel_open_orders_on_stop = str(os.getenv('CANCEL_OPEN_ORDERS_ON_STOP', '0')).strip().lower() in (
    '1', 'true', 'yes', 'y', 'on'
)

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.info("Advertencia: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. Las notificaciones de Telegram no funcionarán.")
    logger.info("Asegúrate de crear un archivo .env con las variables.")
# Clientes de Binance (se inicializan en main() para permitir importación sin tocar red)
# - client_data: MAINNET (solo lectura de mercado: klines/ticker/exchangeInfo)
# - client_trade: TESTNET (solo ejecución: órdenes/balances/cancelaciones)
client_data = None
client_trade = None
# Obtener balance inicial

# Momento de arranque (para ignorar mensajes viejos)
STARTED_AT = int(time.time())

_HTTP_SESSION = requests.Session()

def _http_request(method: str, url: str, *, params=None, data=None, json=None, timeout: float = 10.0,
                  tries: int = 3, backoff_base: float = 1.5, log_prefix: str = "HTTP"):
    for attempt in range(1, tries + 1):
        try:
            resp = _HTTP_SESSION.request(method, url, params=params, data=data, json=json, timeout=timeout)
            # Telegram puede devolver 200 con ok=false, igual devolvemos.
            return resp
        except Exception as e:
            if attempt >= tries:
                raise
            sleep_s = (backoff_base ** (attempt - 1))
            logger.warning(f"{log_prefix}: error intento={attempt}/{tries} url={url}: {e}. Reintentando en {sleep_s:.1f}s")
            time.sleep(sleep_s)

def _binance_call(fn, *args, tries: int = 3, backoff_base: float = 1.7, **kwargs):
    _require_clients()
    for attempt in range(1, tries + 1):
        try:
            out = fn(*args, **kwargs)
            try:
                if SYSTEM_HEALTH_MONITOR is not None:
                    SYSTEM_HEALTH_MONITOR.record_success()
            except Exception:
                pass
            return out
        except BinanceAPIException as be:
            code = getattr(be, 'code', None)
            msg = getattr(be, 'message', str(be))
            # no reintentar errores típicamente definitivos
            # -2015: Invalid API-key, IP, or permissions for action
            if str(code) in ('-2015',):
                try:
                    if SYSTEM_HEALTH_MONITOR is not None:
                        SYSTEM_HEALTH_MONITOR.record_critical(
                            reason="BINANCE_API_EXCEPTION_-2015",
                            meta={"fn": getattr(fn, '__name__', str(fn)), "code": str(code), "msg": str(msg)},
                        )
                except Exception:
                    pass
                raise

            if str(code) in ('-2010', '2010') or 'LOT_SIZE' in str(msg) or 'MIN_NOTIONAL' in str(msg):
                raise
            if attempt >= tries:
                try:
                    if SYSTEM_HEALTH_MONITOR is not None and getattr(fn, '__name__', '') == 'get_account':
                        SYSTEM_HEALTH_MONITOR.record_critical(
                            reason="GET_ACCOUNT_FAILED",
                            meta={"fn": "get_account", "code": str(code), "msg": str(msg)},
                        )
                except Exception:
                    pass
                raise
            sleep_s = (backoff_base ** (attempt - 1))
            logger.warning(f"Binance: error intento={attempt}/{tries} code={code} msg={msg}. Reintentando en {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as e:
            if attempt >= tries:
                try:
                    if SYSTEM_HEALTH_MONITOR is not None and getattr(fn, '__name__', '') == 'get_account':
                        SYSTEM_HEALTH_MONITOR.record_critical(
                            reason="GET_ACCOUNT_FAILED",
                            meta={"fn": "get_account", "error": str(e)},
                        )
                except Exception:
                    pass
                raise
            sleep_s = (backoff_base ** (attempt - 1))
            logger.warning(f"Binance: error intento={attempt}/{tries} {e}. Reintentando en {sleep_s:.1f}s")
            time.sleep(sleep_s)

def _require_clients():
    if client_data is None or client_trade is None:
        raise RuntimeError("Binance clients no inicializados. Ejecuta el script vía __main__.")

def _validate_required_env():
    missing = []
    # Trading (TESTNET) credentials are mandatory for order execution.
    if not (os.getenv('BINANCE_TRADE_API_KEY') or os.getenv('BINANCE_API_KEY')):
        missing.append('BINANCE_TRADE_API_KEY (or BINANCE_API_KEY)')
    if not (os.getenv('BINANCE_TRADE_API_SECRET') or os.getenv('BINANCE_API_SECRET')):
        missing.append('BINANCE_TRADE_API_SECRET (or BINANCE_API_SECRET)')
    if missing:
        raise RuntimeError(f"Faltan variables de entorno requeridas: {', '.join(missing)}")

def _handle_stop_signal(signum, frame):
    logger.warning(f"Señal recibida ({signum}). Deteniendo bot...")
    STOP_EVENT.set()

def init_telegram_offset(url, chat_id=None):
    """
    Drena updates pendientes y devuelve el último update_id para arrancar desde el siguiente.
    """
    try:
        resp = _http_request("GET", url, params={"timeout": 1, "limit": 100}, timeout=5, tries=2, log_prefix="Telegram getUpdates")
        data = resp.json()
        results = data.get("result", [])
        # Opcional: filtrar por chat_id
        if chat_id:
            results = [u for u in results if str(u.get("message", {}).get("chat", {}).get("id")) == str(chat_id)]
        if results:
            return results[-1]["update_id"]
    except Exception as e:
        logger.warning(f"init_telegram_offset: {e}")
    return None

def get_balance():
    try:
        balance = _binance_call(client_trade.get_asset_balance, asset='USDT')
        if not balance:
            return 0.0
        return float(balance.get('free', 0.0))
    except Exception as e:
        logger.error(f"Error obteniendo balance USDT: {e}")
        return 0.0

# --- Posiciones abiertas (PostgreSQL single source of truth) ---
def load_positions(symbol: str) -> list[dict]:
    """Carga posiciones abiertas desde PostgreSQL.

    Si hay un fallo temporal de DB, devuelve el último snapshot en memoria para no romper
    el flujo del bot (pero seguirá reintentando por reconexión automática).
    """
    sym = str(symbol).upper().strip()
    try:
        _require_repos()
        rows = OPEN_POS_REPO.list_by_symbol(sym)  # type: ignore[union-attr]
        if rows is None:
            return list(POSITIONS_CACHE_BY_SYMBOL.get(sym, []))
        POSITIONS_CACHE_BY_SYMBOL[sym] = list(rows)
        return rows
    except Exception as e:
        logger.error(f"[{sym}] load_positions(DB) falló: {e}")
        return list(POSITIONS_CACHE_BY_SYMBOL.get(sym, []))


def save_positions(symbol: str, positions_list: list[dict]) -> None:
    """Reconcilia posiciones para el símbolo en PostgreSQL (INSERT/UPDATE/DELETE).

    Mantiene compatibilidad con el patrón legacy: load -> mutate -> save.
    """
    sym = str(symbol).upper().strip()
    POSITIONS_CACHE_BY_SYMBOL[sym] = list(positions_list or [])
    try:
        _require_repos()
        ok = OPEN_POS_REPO.replace_positions(sym, positions_list or [])  # type: ignore[union-attr]
        if not ok:
            logger.warning(f"[{sym}] save_positions(DB) no pudo persistir (continuando).")
    except Exception as e:
        logger.error(f"[{sym}] save_positions(DB) falló: {e}")


def load_all_open_positions() -> list[dict]:
    try:
        _require_repos()
        rows = OPEN_POS_REPO.list_all()  # type: ignore[union-attr]
        return list(rows or [])
    except Exception as e:
        logger.warning("load_all_open_positions falló: %s", e)
        return []

# Obtener datos en tiempo real
def get_data_binance(symbol, interval='1h', limit=40):
    """
    Devuelve DataFrame con las últimas `limit` velas para `symbol`.
    Usa interval como string ('1h', '4h', '15m', etc.) para evitar problemas con constantes.
    """
    if DB is None:
        raise RuntimeError("DB no inicializada. `USE_DATABASE=true` es obligatorio y la conexión debe estar lista.")

    # ENTRADAS: SOLO desde PostgreSQL (fuente única de verdad para histórico).
    # La ingesta/backfill se hace en un proceso separado (Market Data Process).
    return read_klines_df(
        db=DB,
        symbol=str(symbol).upper().strip(),
        interval=str(interval).strip(),
        limit=int(limit),
    )

#Calculo de metricas tecnicas para el trading

def cal_metrics_technig(df, rsi_w, sma_short_w, sma_long_w):

    
    # Calculo del RSI (Relative Strength Index) 
    # Evalúa si el activo está sobrecomprado o sobrevendido.
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_w).rsi()

    # Cálculo del Stochastic RSI cal_metrics_technig(data, 14, 10, 20)
    stochrsi = ta.momentum.StochRSIIndicator(df['close'], window=rsi_w, smooth1=3, smooth2=3)
    df['stochrsi_k'] = stochrsi.stochrsi_k()
    df['stochrsi_d'] = stochrsi.stochrsi_d()

    # Calculo de la Media Móvil Simple (SMA)
    # Identifica tendencias generales del mercado.
    df['sma_short'] = ta.trend.SMAIndicator(df['close'], window=sma_short_w).sma_indicator()
    df['sma_long'] = ta.trend.SMAIndicator(df['close'], window=sma_long_w).sma_indicator()

    # Calculo de MACD (Moving Average Convergence Divergence)
    # Ayuda a determinar la fuerza y dirección de la tendencia.
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()

    # Métricas extra para detección de régimen de mercado
    # EMA200 (tendencia de largo plazo)
    try:
        ema200 = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        # Si no hay suficientes velas, la librería devuelve NaN hasta completar la ventana.
        if len(ema200) > 0 and pd.isna(ema200.iloc[-1]):
            raise ValueError("EMA200 not ready (NaN at latest)")
        df['ema200'] = ema200
    except Exception:
        # Fallback: calcula EMA con ewm para poder operar/depurar incluso con historial corto (p.ej. testnet)
        if df is not None and not df.empty and 'close' in df.columns:
            _log_throttled(
                "ema200_fallback",
                logging.WARNING,
                "EMA200 no disponible con el historial actual (se requieren 200 velas). Usando fallback ewm; velas=%s. "
                "Si estás en BINANCE_TESTNET, es normal que el histórico sea limitado.",
                int(len(df)),
                every_s=600.0,
            )
            df['ema200'] = df['close'].ewm(span=200, adjust=False, min_periods=1).mean()
        else:
            df['ema200'] = pd.NA

    # EMA50 (tendencia intermedia para filtro de entradas)
    try:
        ema50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        if len(ema50) > 0 and pd.isna(ema50.iloc[-1]):
            raise ValueError("EMA50 not ready (NaN at latest)")
        df['ema50'] = ema50
    except Exception:
        if df is not None and not df.empty and 'close' in df.columns:
            df['ema50'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()
        else:
            df['ema50'] = pd.NA

    # ATR (volatilidad) para TP/SL dinámicos
    try:
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_WINDOW)
        df['atr'] = atr_indicator.average_true_range()
    except Exception:
        df['atr'] = pd.NA

    # ADX (fuerza de tendencia)
    try:
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_indicator.adx()
    except Exception:
        df['adx'] = pd.NA

    return df


def calculate_atr_levels(buy_price: float, atr: float, tp_multiplier: float, sl_multiplier: float) -> tuple[float, float]:
    """Calcula niveles dinámicos por ATR para TP/SL."""
    tp = buy_price + (atr * tp_multiplier)
    sl = buy_price - (atr * sl_multiplier)
    return tp, sl


def update_trailing_stop_atr(position: dict, close_price: float, atr: float, trailing_sl_multiplier: float) -> bool:
    """Actualiza max_price y stop_loss por ATR sin mover stop hacia abajo.

    Reglas:
    - max_price solo sube
    - stop_loss nunca baja
    - trailing requiere position['trailing_active'] == True
    """
    if not position.get('trailing_active'):
        return False

    buy_price = float(position.get('buy_price', 0.0) or 0.0)
    prev_max = float(position.get('max_price', buy_price) or buy_price)
    new_max = max(prev_max, float(close_price))
    position['max_price'] = new_max

    try:
        current_sl = float(position.get('stop_loss', 0.0) or 0.0)
    except Exception:
        current_sl = 0.0

    new_stop = new_max - (atr * trailing_sl_multiplier)
    # Nunca mover stop hacia abajo
    if current_sl and new_stop < current_sl:
        return False
    position['stop_loss'] = max(current_sl, new_stop) if current_sl else new_stop
    return True


def _utc_day_key(ts: datetime | pd.Timestamp | None = None) -> date:
    if ts is None:
        return datetime.now(timezone.utc).date()
    if isinstance(ts, pd.Timestamp):
        # pandas timestamp puede venir sin tz
        dt = ts.to_pydatetime()
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


@dataclass
class SymbolRiskState:
    day: date
    day_start_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    peak_value: float = 0.0
    max_daily_drawdown: float = 0.0
    disabled_until_ts: float = 0.0

    # Métricas instantáneas (observabilidad)
    committed_capital: float = 0.0
    current_value: float = 0.0
    floating_pnl: float = 0.0
    floating_loss: float = 0.0
    current_drawdown: float = 0.0


class RiskManager:
    """Gestor de riesgo global por símbolo.

    Se mantiene en memoria (thread-safe) y aplica reglas duras para bloquear entradas.
    """

    def __init__(self, symbols: list[str]):
        self._lock = threading.Lock()
        today = _utc_day_key()
        self._state: dict[str, SymbolRiskState] = {
            s: SymbolRiskState(day=today) for s in symbols
        }

    def _roll_day_if_needed(self, symbol: str, today: date, equity_total: float | None = None):
        st = self._state[symbol]
        if st.day != today:
            self._state[symbol] = SymbolRiskState(day=today, day_start_equity=float(equity_total or 0.0))

    def observe(self, symbol: str, *, positions: list[dict], price: float, equity_total: float, now_ts: float | None = None):
        """Actualiza métricas de riesgo intradía (peak y drawdown) para el símbolo."""
        now_ts = float(now_ts or time.time())
        today = _utc_day_key(datetime.fromtimestamp(now_ts, tz=timezone.utc))

        committed = 0.0
        current_value = 0.0
        for p in positions or []:
            try:
                bp = float(p.get('buy_price', 0.0) or 0.0)
                amt = float(p.get('amount', 0.0) or 0.0)
            except Exception:
                continue
            committed += (bp * amt)
            current_value += (float(price) * amt)

        with self._lock:
            self._roll_day_if_needed(symbol, today, equity_total)
            st = self._state[symbol]
            if st.day_start_equity <= 0:
                st.day_start_equity = float(equity_total or 0.0)

            st.committed_capital = float(committed)
            st.current_value = float(current_value)
            st.floating_pnl = float(current_value - committed)
            st.floating_loss = float(max(0.0, committed - current_value))

            st.peak_value = max(st.peak_value, current_value)
            if st.peak_value > 0:
                dd = max(0.0, (st.peak_value - current_value) / st.peak_value)
                st.max_daily_drawdown = max(st.max_daily_drawdown, dd)
                st.current_drawdown = dd
            else:
                st.current_drawdown = 0.0

    def record_realized_pnl(self, symbol: str, pnl: float, when: datetime | pd.Timestamp | None = None, *, equity_total: float | None = None):
        """Registra PnL realizado y aplica regla de pausa 24h por pérdida diaria."""
        when_dt = when
        if when_dt is None:
            when_dt = datetime.now(timezone.utc)
        today = _utc_day_key(when_dt)
        now_ts = time.time()
        with self._lock:
            self._roll_day_if_needed(symbol, today, equity_total)
            st = self._state[symbol]
            st.daily_realized_pnl += float(pnl or 0.0)

            base = st.day_start_equity if st.day_start_equity > 0 else float(equity_total or 0.0)
            if base > 0:
                loss_frac = max(0.0, (-st.daily_realized_pnl) / base)
                if loss_frac >= DAILY_LOSS_LIMIT_FRAC:
                    st.disabled_until_ts = max(st.disabled_until_ts, now_ts + SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS)

    def can_open(self, symbol: str, *, equity_total: float, committed_capital: float, symbol_drawdown: float, now_ts: float | None = None) -> tuple[bool, str | None]:
        now_ts = float(now_ts or time.time())
        today = _utc_day_key(datetime.fromtimestamp(now_ts, tz=timezone.utc))
        with self._lock:
            self._roll_day_if_needed(symbol, today, equity_total)
            st = self._state[symbol]

            if st.disabled_until_ts and now_ts < st.disabled_until_ts:
                remaining = int(st.disabled_until_ts - now_ts)
                return False, f"símbolo pausado por pérdida diaria ({remaining}s restantes)"

            # Regla: no abrir nuevas posiciones si drawdown actual del símbolo supera umbral
            if symbol_drawdown > MAX_SYMBOL_DRAWDOWN_FRAC:
                return False, f"drawdown del símbolo {symbol_drawdown*100:.2f}% > {MAX_SYMBOL_DRAWDOWN_FRAC*100:.2f}%"

            # Regla: no comprometer demasiado del equity total
            if equity_total > 0 and committed_capital > (equity_total * MAX_CAPITAL_COMMITTED_FRAC):
                frac = committed_capital / equity_total
                return False, f"capital comprometido {frac*100:.2f}% > {MAX_CAPITAL_COMMITTED_FRAC*100:.2f}%"

            return True, None

    def snapshot(self, symbol: str, now_ts: float | None = None) -> dict:
        """Snapshot thread-safe del estado del símbolo."""
        now_ts = float(now_ts or time.time())
        today = _utc_day_key(datetime.fromtimestamp(now_ts, tz=timezone.utc))
        with self._lock:
            # si cambia el día, el snapshot refleja el reset
            self._roll_day_if_needed(symbol, today, None)
            st = self._state[symbol]
            return {
                'day': st.day,
                'day_start_equity': st.day_start_equity,
                'daily_realized_pnl': st.daily_realized_pnl,
                'peak_value': st.peak_value,
                'max_daily_drawdown': st.max_daily_drawdown,
                'disabled_until_ts': st.disabled_until_ts,
                'committed_capital': st.committed_capital,
                'current_value': st.current_value,
                'floating_pnl': st.floating_pnl,
                'floating_loss': st.floating_loss,
                'current_drawdown': st.current_drawdown,
            }


def _positions_cost_basis(positions: list[dict]) -> float:
    total = 0.0
    for p in positions or []:
        try:
            bp = float(p.get('buy_price', 0.0) or 0.0)
            amt = float(p.get('amount', 0.0) or 0.0)
        except Exception:
            continue
        total += (bp * amt)
    return total


def _positions_current_value(positions: list[dict], price: float) -> float:
    total = 0.0
    for p in positions or []:
        try:
            amt = float(p.get('amount', 0.0) or 0.0)
        except Exception:
            continue
        total += (float(price) * amt)
    return total


def compute_equity_total(symbols: list[str], locks: dict | None = None, price_overrides: dict | None = None) -> float:
    """Equity total aproximado en USDT: balance libre + valor actual de posiciones abiertas.

    Se usa solo cuando se evalúa abrir nuevas posiciones (no en cada tick) para minimizar llamadas.
    """
    usdt_free = get_balance()
    total_positions_value = 0.0

    price_overrides = price_overrides or {}
    locks = locks or {}

    for sym in symbols:
        try:
            lock = locks.get(sym) if isinstance(locks, dict) else None
            if lock:
                with lock:
                    pos = load_positions(sym)
            else:
                pos = load_positions(sym)
            if not pos:
                continue
            px = price_overrides.get(sym)
            if px is None:
                px = get_precio_actual(sym)
            if px is None:
                continue
            total_positions_value += _positions_current_value(pos, float(px))
        except Exception:
            continue

    return float(usdt_free) + float(total_positions_value)


def equity_snapshot_worker(symbols: list[str], locks: dict[str, threading.Lock] | None = None):
    """Persiste snapshots periódicos de equity para analítica/drawdowns.

    - Corre en thread separado.
    - No bloquea trading (usa locks solo para leer un snapshot consistente por símbolo).
    - Si la DB falla, se loguea y se reintenta en el próximo tick.
    """
    logger.info("Equity snapshots thread iniciado. intervalo=%ss", EQUITY_SNAPSHOT_INTERVAL)
    while not STOP_EVENT.is_set():
        try:
            _require_clients()
            _require_repos()

            now_ts = datetime.now(timezone.utc).replace(microsecond=0)
            usdt_free = float(get_balance())

            positions_json: dict[str, list[dict]] = {}
            positions_value = 0.0

            for sym in symbols:
                sym_u = str(sym).upper().strip()
                lock = (locks or {}).get(sym_u) if isinstance(locks, dict) else None
                if lock:
                    with lock:
                        pos = load_positions(sym_u)
                else:
                    pos = load_positions(sym_u)

                positions_json[sym_u] = list(pos or [])
                if not pos:
                    continue
                px = get_precio_actual(sym_u)
                if px is None:
                    continue
                positions_value += _positions_current_value(pos, float(px))

            equity_total = float(usdt_free) + float(positions_value)

            ok = EQUITY_REPO.insert_snapshot(  # type: ignore[union-attr]
                timestamp=now_ts,
                equity_total=equity_total,
                usdt_balance=usdt_free,
                positions_value=positions_value,
                positions_json=positions_json,
            )
            if not ok:
                logger.warning("Equity snapshot no persistido (DB error).")
            else:
                # Global equity kill switch (Risk Layer v2)
                try:
                    if GLOBAL_RISK_CONTROLLER is not None:
                        GLOBAL_RISK_CONTROLLER.on_equity_snapshot(equity_total=float(equity_total), when=now_ts)
                except Exception as e:
                    logger.warning("GlobalRiskController error: %s", e)

        except Exception as e:
            logger.error(f"Equity snapshot worker error: {e}")

        STOP_EVENT.wait(max(1, int(EQUITY_SNAPSHOT_INTERVAL)))

# Función para guardar logs de operaciones para análisis ML (PostgreSQL)

def _normalize_trade_reason(description: str | None) -> str | None:
    if not description:
        return None
    d = str(description).strip().upper().replace(" ", "_")
    if d in ("TAKE_PROFIT", "TP"):
        return "TAKE_PROFIT"
    if d in ("STOP_LOSS", "SL"):
        return "STOP_LOSS"
    if d in ("SELL_ALL", "SELLALL", "SELL-ALL"):
        return "SELL_ALL"
    # Fallback: conservar un motivo corto si entra en el esquema
    if len(d) <= 20:
        return d
    return d[:20]


def log_trade(timestamp, symbol, trade_type, price, amount, profit, volatility, rsi, stochrsi_k, stochrsi_d, description, extra: dict | None = None):
    """Inserta trade en trading.trade_history.

    trade_type: 'buy'/'sell' (case-insensitive)
    description: BUY / TAKE PROFIT / STOP LOSS / SELL ALL (se normaliza a reason)
    extra: puede incluir buy_price/sell_price/regime
    """
    extra = extra or {}
    sym = str(symbol).upper().strip()
    side = str(trade_type).strip().upper()
    if side == 'BUY':
        side = 'BUY'
    elif side == 'SELL':
        side = 'SELL'
    elif side.lower() == 'buy':
        side = 'BUY'
    else:
        side = 'SELL'

    buy_price = extra.get('buy_price') if side == 'SELL' else price
    sell_price = price if side == 'SELL' else None

    realized_pnl = float(profit or 0.0) if side == 'SELL' else 0.0
    realized_pnl_pct = None
    try:
        if side == 'SELL':
            bp = float(buy_price) if buy_price is not None else None
            amt = float(amount) if amount is not None else None
            if bp and amt and (bp * amt) != 0:
                realized_pnl_pct = (realized_pnl / (bp * amt)) * 100.0
    except Exception:
        realized_pnl_pct = None

    try:
        _require_repos()
        TRADE_REPO.insert_trade(  # type: ignore[union-attr]
            symbol=sym,
            side=side,
            reason=_normalize_trade_reason(description),
            buy_price=buy_price,
            sell_price=sell_price,
            amount=amount,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            volatility=volatility,
            rsi=rsi,
            stochrsi_k=stochrsi_k,
            stochrsi_d=stochrsi_d,
            regime=extra.get('regime'),
            executed_at=timestamp,
        )
    except Exception as e:
        logger.error(f"[{sym}] log_trade(DB) falló: {e}")

# Función para enviar notificaciones a Telegram
def send_positions_to_telegram(symbol, data, token, chat_id):
    if not token or not chat_id:
        logger.info("Telegram no configurado; omitiendo /posiciones")
        return
    positions = load_positions(symbol)
    if not positions:
        message = "No hay posiciones abiertas actualmente. \n"
        # Obtener el último precio y métricas técnicas
        df = cal_metrics_technig(data, 14, 10, 20)
        close_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        stochrsi_k = df['stochrsi_k'].iloc[-1]
        stochrsi_d = df['stochrsi_d'].iloc[-1]
        message += (
            f"\nPrecio actual: {close_price:.2f} {symbol}\n"
            f"RSI: {rsi:.2f}\n"
            f"StochRSI K: {stochrsi_k:.2f}\n"
            f"StochRSI D: {stochrsi_d:.2f}\n"
            f"\n*Criterio de compra (nuevo):* precio > EMA200, EMA50 > EMA200 y RSI entre {RSI_CONFIRM_MIN:.0f}-{RSI_CONFIRM_MAX:.0f}\n"
            f"(En régimen BEAR no se abren nuevas posiciones)\n"
        )
        message += "\n\nEl balance actual es: {:.2f} USDT".format(get_balance())
        message += "\n\n*Esperando nuevas oportunidades de compra...*"
        message += "\n\n*¡Mantente atento a las actualizaciones!*"
    else:
        message = f"📊 *Posiciones abiertas de {symbol}:*\n"
        for pos in positions:
            message += (
                f"- Precio compra: {pos['buy_price']:.2f} USDT\n"
                f"  Cantidad: {pos['amount']:.6f} {symbol}\n"
                f"  Fecha: {pos['timestamp']}\n"
            )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        _http_request("POST", url, data=payload, timeout=10, tries=3, log_prefix="Telegram sendMessage")
    except Exception as e:
        logger.error(f"Error enviando mensaje a Telegram: {e}")

# Función para enviar notificaciones a Telegram
def send_event_to_telegram(message, token, chat_id):
    if not token or not chat_id:
        logger.info("Telegram no configurado; evento: %s", message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        _http_request("POST", url, data=payload, timeout=10, tries=3, log_prefix="Telegram sendMessage")
    except Exception as e:
        logger.error(f"Error enviando mensaje a Telegram: {e}")
# Función para calcular el % de cambio en los últimos 5 intervalos de 4h
def market_change_last_5_intervals(symbol):
    """
    Calcula el % de cambio en cada uno de los últimos 5 intervalos de 4h.
    Devuelve una lista con el % de cambio por intervalo y el promedio total.
    """
    # Obtener las últimas 5 velas de 4h (persistidas; sincroniza desde MAINNET si falta data)
    df = get_data_binance(symbol, interval='4h', limit=5)
    if df is None or df.empty:
        return 0.0
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # En vez de sumar retornos por vela (puede distorsionar), calcula el retorno acumulado
    # desde la primera apertura hasta el último cierre.
    if df.empty:
        return 0.0
    first_open = float(df['open'].iloc[0])
    last_close = float(df['close'].iloc[-1])
    if first_open == 0:
        return 0.0
    cumulative_change = ((last_close - first_open) / first_open) * 100
    return cumulative_change
# Obtener precio actual directamente desde Binance
def get_precio_actual(symbol):
    try:
        # SALIDAS/monitoreo: precio en tiempo real desde TESTNET (evita divergencias de precio con MAINNET).
        ticker = _binance_call(client_trade.get_symbol_ticker, symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.error(f"Error obteniendo precio para {symbol}: {e}")
        return None

# Función para ejecutar órdenes con confirmación y reintentos
def ejecutar_orden_con_confirmacion(
    tipo: str,
    symbol: str,
    cantidad: float,
    max_intentos: int = 5,
    verificaciones_por_intento: int = 6,
    delay_verificacion: float = 2.0,
    backoff_base: float = 2.0,
    permitir_parcial: bool = True,
):
    """
    Envía orden MARKET y espera confirmación consultando get_order.
    Reintenta si no se llena dentro de las verificaciones configuradas.
    Si permitir_parcial=True y la orden queda PARTIALLY_FILLED tras cancelación,
    acepta el parcial y retorna el estado final.
    """
    tipo = tipo.lower()
    if tipo not in ('buy', 'sell'):
        raise ValueError("Tipo de orden inválido. Usa 'buy' o 'sell'.")
    side_fn = client_trade.order_market_buy if tipo == 'buy' else client_trade.order_market_sell

    for intento in range(1, max_intentos + 1):
        try:
            # Enviar orden
            resp = _binance_call(side_fn, symbol=symbol, quantity=cantidad, tries=2)
            order_id = resp['orderId']
            logger.info(f"[{symbol}] 📨 Orden {tipo.upper()} enviada id={order_id} qty={cantidad} intento={intento}")

            filled = False
            last_status = None
            partial_executed = 0.0

            for v in range(1, verificaciones_por_intento + 1):
                time.sleep(delay_verificacion)
                try:
                    status_info = _binance_call(client_trade.get_order, symbol=symbol, orderId=order_id, tries=2)
                except BinanceAPIException as ve:
                    logger.error(f"[{symbol}] Error get_order id={order_id}: {ve}")
                    continue

                status = status_info.get('status')
                executed_qty = float(status_info.get('executedQty', 0) or 0)
                cummulative_quote_qty = float(status_info.get('cummulativeQuoteQty', 0) or 0)
                last_status = status

                if status == 'FILLED':
                    logger.info(f"[{symbol}] ✅ Orden {tipo.upper()} FILLED qty={executed_qty}")
                    send_event_to_telegram(
                        f"✅ {tipo.upper()} {symbol}\nQty: {executed_qty}\nNotional: {cummulative_quote_qty:.8f} USDT",
                        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                    )
                    return status_info

                if status == 'PARTIALLY_FILLED':
                    partial_executed = executed_qty
                    logger.info(f"[{symbol}] ⏳ Parcial {tipo.upper()} {executed_qty}/{cantidad} (verif {v}/{verificaciones_por_intento})")
                    continue  # seguir verificando

                if status in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    logger.warning(f"[{symbol}] ❌ Orden {order_id} estado terminal={status}")
                    # Circuit breaker: rechazos repetidos pueden indicar un problema sistémico
                    try:
                        if SYSTEM_HEALTH_MONITOR is not None and status in ('REJECTED', 'EXPIRED'):
                            SYSTEM_HEALTH_MONITOR.record_critical(
                                reason="ORDER_REJECTED",
                                meta={"symbol": symbol, "order_id": order_id, "status": status},
                            )
                    except Exception:
                        pass
                    break  # salir de verificación e ir a reintento

                # NEW u otros: seguir
                logger.debug(f"[{symbol}] Estado {status} verif {v}/{verificaciones_por_intento}")

            # Si terminó verificación sin FILLED
            if last_status in ('NEW', 'PARTIALLY_FILLED'):
                # Intentar cancelar para evitar que quede colgada
                try:
                    _binance_call(client_trade.cancel_order, symbol=symbol, orderId=order_id, tries=2)
                    logger.info(f"[{symbol}] 🛑 Cancelada orden pendiente id={order_id} status_final={last_status}")
                except Exception as ce:
                    logger.warning(f"[{symbol}] No se pudo cancelar orden {order_id}: {ce}")

                if last_status == 'PARTIALLY_FILLED' and permitir_parcial and partial_executed > 0:
                    logger.info(f"[{symbol}] ✅ Aceptando ejecución parcial qty={partial_executed}")
                    send_event_to_telegram(
                        f"✅ Parcial {tipo.upper()} {symbol} qty={partial_executed} (aceptada)",
                        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                    )
                    status_info['acceptedPartial'] = True
                    return status_info

            # Backoff antes de próximo intento
            sleep_time = backoff_base ** (intento - 1)
            jitter = min(1.0, sleep_time * 0.15)
            time.sleep(sleep_time + (jitter * 0.5))
            logger.warning(f"[{symbol}] Reintento {intento+1}/{max_intentos} para orden {tipo.upper()} (status={last_status})")

        except BinanceAPIException as be:
            code = getattr(be, 'code', 'N/A')
            msg = getattr(be, 'message', str(be))
            logger.error(f"[{symbol}] ❌ BinanceAPIException intento={intento} code={code} msg={msg}")

            # Errores no recuperables
            if str(code) in ('-2010', '2010'):  # fondos insuficientes
                send_event_to_telegram(f"❌ Fondos insuficientes {symbol} {tipo} qty={cantidad}", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                return None
            if 'LOT_SIZE' in msg or 'MIN_NOTIONAL' in msg:
                send_event_to_telegram(f"❌ Filtro LOT_SIZE/MIN_NOTIONAL {symbol} qty={cantidad}", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                return None

            time.sleep(1.5)

        except Exception as e:
            logger.error(f"[{symbol}] ❌ Error genérico en {tipo.upper()} intento={intento}: {e}")
            time.sleep(1.0)

    alerta = f"❌ No se pudo completar {tipo.upper()} {symbol} qty={cantidad} tras {max_intentos} intentos."
    logger.error(alerta)
    send_event_to_telegram(alerta, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    return None


# Función para monitorear stop loss de forma asíncrona
# Esta función se ejecutará en un hilo separado para monitorear las posiciones abiertas y ejecutar stop
def monitoring_open_position(symbol, lock):
    logger.info(f"[{symbol}] Iniciando hilo monitor posiciones")
    _require_clients()

    # Cache de filtros por símbolo (reduce llamadas y permite mejores logs)
    try:
        symbol_filters = get_symbol_filters(symbol)
    except Exception as e:
        logger.warning(f"[{symbol}] No se pudieron obtener filtros de símbolo (monitor): {e}")
        symbol_filters = None

    while not STOP_EVENT.is_set():
        try:
            # 1) Cargar posiciones rápido bajo lock
            acquired = lock.acquire(timeout=1.0)
            if not acquired:
                # si está ocupado, intenta más tarde
                time.sleep(2)
                continue
            try:
                positions = load_positions(symbol)
            finally:
                lock.release()

            if not positions:
                STOP_EVENT.wait(30)
                continue

            # 2) Fuera del lock: llamadas de red y cálculo
            # Pedir histórico suficiente para EMA200/ATR/ADX (si no, EMA200 será NaN y caerá al fallback ewm)
            data = get_data_binance(symbol, interval=timeframe, limit=260)
            df_metrics = cal_metrics_technig(data.copy(), 14, 10, 20)
            # Precio para ejecución/SL/TP: usar Binance live para evitar retrasos por DB.
            live_price = get_precio_actual(symbol)
            close_price = df_metrics['close'].iloc[-1]
            current_price = float(live_price) if live_price is not None else float(close_price)
            high = df_metrics['high'].iloc[-1]
            low = df_metrics['low'].iloc[-1]
            open_price = df_metrics['open'].iloc[-1]
            volatility = (high - low) / open_price * 100 if open_price != 0 else 0
            rsi = df_metrics['rsi'].iloc[-1] if 'rsi' in df_metrics.columns else None
            stochrsi_k = df_metrics['stochrsi_k'].iloc[-1] if 'stochrsi_k' in df_metrics.columns else None
            stochrsi_d = df_metrics['stochrsi_d'].iloc[-1] if 'stochrsi_d' in df_metrics.columns else None
            timestamp = pd.Timestamp.now()

            # ATR basado en última vela cerrada para evitar lookahead
            try:
                atr_now = df_metrics['atr'].iloc[-2]
                atr_now = float(atr_now) if not pd.isna(atr_now) else None
            except Exception:
                atr_now = None

            # Equity estimado para risk manager (USDT libre + valor posiciones del símbolo)
            try:
                equity_est = float(get_balance()) + _positions_current_value(positions, float(current_price))
            except Exception:
                equity_est = float(get_balance())

            # Alimentar risk manager con métricas intradía
            try:
                RISK_MANAGER.observe(symbol, positions=positions, price=float(current_price), equity_total=float(equity_est))
            except Exception:
                pass

            # 3) Preparar actualizaciones y ventas fuera del lock
            to_update = []
            to_sell = []

            for position in positions:
                # Compatibilidad: posiciones antiguas pueden no tener nuevas claves
                try:
                    buy_price = float(position.get('buy_price', 0.0) or 0.0)
                    amount = float(position.get('amount', 0.0) or 0.0)
                except Exception:
                    continue

                try:
                    take_profit = float(position.get('take_profit', buy_price * (1 + take_profit_pct / 100)) or 0.0)
                except Exception:
                    take_profit = buy_price * (1 + take_profit_pct / 100)

                # TP inicial (para activar trailing) - si no existe, usa take_profit actual
                try:
                    tp_initial = float(position.get('tp_initial', take_profit) or take_profit)
                except Exception:
                    tp_initial = take_profit

                try:
                    stop_loss = float(position.get('stop_loss', buy_price * (1 - stop_loss_pct / 100)) or 0.0)
                except Exception:
                    stop_loss = buy_price * (1 - stop_loss_pct / 100)

                # Activación de trailing solo tras superar TP inicial
                trailing_active = bool(position.get('trailing_active', False))
                if (not trailing_active) and current_price >= tp_initial:
                    trailing_active = True
                    to_update.append({
                        'match': {'id': position.get('id'), 'buy_price': buy_price, 'timestamp': str(position.get('timestamp'))},
                        'fields': {
                            'trailing_active': True,
                            'tp_initial': tp_initial,
                            'max_price': max(float(position.get('max_price', buy_price) or buy_price), float(current_price)),
                        }
                    })

                effective_stop_loss = stop_loss

                # Trailing stop por ATR (preferido). Si no hay ATR disponible, cae al trailing % anterior.
                if trailing_active:
                    updated_fields = {}
                    if atr_now is not None and atr_now > 0:
                        # multiplicador por posición si existe; si no, por régimen guardado o default LATERAL
                        try:
                            trailing_mult = float(position.get('trailing_sl_atr_mult', 0) or 0)
                        except Exception:
                            trailing_mult = 0.0
                        if trailing_mult <= 0:
                            pos_regime = str(position.get('regime') or 'LATERAL').upper()
                            trailing_mult = float((ATR_MULTIPLIERS.get(pos_regime) or ATR_MULTIPLIERS['LATERAL'])['trailing_sl'])

                        tmp = dict(position)
                        tmp['trailing_active'] = True
                        tmp.setdefault('max_price', float(position.get('max_price', buy_price) or buy_price))
                        changed = update_trailing_stop_atr(tmp, float(current_price), float(atr_now), float(trailing_mult))
                        if changed:
                            updated_fields['max_price'] = tmp.get('max_price')
                            updated_fields['stop_loss'] = tmp.get('stop_loss')
                            updated_fields['trailing_active'] = True
                            updated_fields['tp_initial'] = tp_initial
                    else:
                        # Fallback: trailing % antiguo (solo si no hay ATR)
                        try:
                            trailing_sl_pct_pos = float(position.get('trailing_sl_pct', trailing_stop_pct))
                        except Exception:
                            trailing_sl_pct_pos = trailing_stop_pct
                        # stop nunca baja
                        candidate = float(current_price) * (1 - trailing_sl_pct_pos / 100)
                        if candidate > stop_loss:
                            updated_fields['stop_loss'] = candidate
                            updated_fields['max_price'] = max(float(position.get('max_price', buy_price) or buy_price), float(current_price))
                            updated_fields['trailing_active'] = True
                            updated_fields['tp_initial'] = tp_initial

                    if updated_fields:
                        if 'stop_loss' in updated_fields:
                            try:
                                effective_stop_loss = float(updated_fields['stop_loss'])
                            except Exception:
                                effective_stop_loss = stop_loss
                        to_update.append({
                            'match': {'id': position.get('id'), 'buy_price': buy_price, 'timestamp': str(position.get('timestamp'))},
                            'fields': updated_fields
                        })

                # Venta por stop
                # Nota: stop_loss puede ser actualizado bajo lock; aquí usamos el valor actual.
                if current_price <= effective_stop_loss:
                    to_sell.append({
                        'position': position,
                        'reason': 'TAKE PROFIT' if effective_stop_loss >= buy_price else 'STOP LOSS'
                    })

            # 4) Aplicar actualizaciones rápidas bajo lock
            if to_update:
                acquired = lock.acquire(timeout=1.0)
                if acquired:
                    try:
                        current = load_positions(symbol)
                        changed = False
                        for upd in to_update:
                            pid = upd['match'].get('id')
                            bp = float(upd['match']['buy_price'])
                            ts = str(upd['match']['timestamp'])
                            for p in current:
                                if pid is not None and p.get('id') is not None and int(p.get('id')) == int(pid):
                                    p.update(upd['fields'])
                                    changed = True
                                    break
                                if float(p.get('buy_price', 0)) == bp and str(p.get('timestamp')) == ts:
                                    p.update(upd['fields'])
                                    changed = True
                        if changed:
                            save_positions(symbol, current)
                    finally:
                        lock.release()

            # 5) Procesar ventas: quitar posición bajo lock, vender fuera del lock
            for item in to_sell:
                pos = item['position']
                reason = item['reason']

                # Remover del archivo bajo lock
                removed = None
                acquired = lock.acquire(timeout=1.0)
                if acquired:
                    try:
                        current = load_positions(symbol)
                        for p in list(current):
                            if pos.get('id') is not None and p.get('id') is not None and int(p.get('id')) == int(pos.get('id')):
                                current.remove(p)
                                removed = p
                                break
                            if float(p.get('buy_price', 0)) == float(pos['buy_price']) and str(p.get('timestamp')) == str(pos['timestamp']):
                                current.remove(p)
                                removed = p
                                break
                        if removed:
                            save_positions(symbol, current)
                    finally:
                        lock.release()

                if not removed:
                    continue  # ya fue procesada por otro hilo

                # Fuera del lock: validar qty y vender
                bal = get_base_asset_balance(symbol)
                free_qty = float(bal.get('free', 0.0) or 0.0)
                locked_qty = float(bal.get('locked', 0.0) or 0.0)

                requested_qty = float(removed.get('amount', 0.0) or 0.0)
                sell_qty_raw = min(requested_qty, free_qty)
                sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, current_price, for_sell=True, filters=symbol_filters)

                # Si el balance está bloqueado en órdenes abiertas, un STOP LOSS puede fallar.
                if sell_qty is None and cancel_open_orders_on_stop and locked_qty > 0:
                    canceled = cancel_all_open_orders(symbol)
                    if canceled > 0:
                        bal = get_base_asset_balance(symbol)
                        free_qty = float(bal.get('free', 0.0) or 0.0)
                        locked_qty = float(bal.get('locked', 0.0) or 0.0)
                        sell_qty_raw = min(requested_qty, free_qty)
                        sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, current_price, for_sell=True, filters=symbol_filters)

                if sell_qty is None:
                    min_qty_dbg = (symbol_filters or {}).get('min_qty')
                    step_dbg = (symbol_filters or {}).get('step_size')
                    logger.warning(
                        f"[{symbol}] No se puede vender ({reason}) requested={requested_qty} free={free_qty} locked={locked_qty} "
                        f"raw={sell_qty_raw} min_qty={min_qty_dbg} step={step_dbg} motivo={motivo}"
                    )
                    send_event_to_telegram(
                        f"⚠️ No se puede vender {symbol} ({reason})\n"
                        f"requested={requested_qty} free={free_qty} locked={locked_qty}\n"
                        f"raw={sell_qty_raw} motivo={motivo}",
                        TELEGRAM_BOT_TOKEN,
                        TELEGRAM_CHAT_ID,
                    )
                    # opcional: reinsertar posición si no se vendió
                    acquired = lock.acquire(timeout=1.0)
                    if acquired:
                        try:
                            current = load_positions(symbol)
                            current.append(removed)
                            save_positions(symbol, current)
                        finally:
                            lock.release()
                    continue

                resp = ejecutar_orden_con_confirmacion('sell', symbol, sell_qty)
                if not resp:
                    logger.error(f"[{symbol}] Venta fallida qty={sell_qty} ({reason})")
                    # opcional: reinsertar posición si falla la orden
                    acquired = lock.acquire(timeout=1.0)
                    if acquired:
                        try:
                            current = load_positions(symbol)
                            current.append(removed)
                            save_positions(symbol, current)
                        finally:
                            lock.release()
                    continue

                emoji = '🎯' if reason == 'TAKE PROFIT' else '🚨'
                msg = f'{emoji} {reason}: Vendemos {sell_qty:.6f} {symbol} a {current_price:.4f} USDT'
                send_event_to_telegram(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                # registrar pnl realizado en risk manager
                realized = (current_price - float(removed.get('buy_price', 0.0) or 0.0)) * float(sell_qty)
                try:
                    RISK_MANAGER.record_realized_pnl(symbol, realized, timestamp, equity_total=equity_est)
                except Exception:
                    pass
                log_trade(
                    timestamp, symbol, 'sell', current_price, sell_qty,
                    (current_price - removed['buy_price']) * sell_qty,
                    volatility, rsi, stochrsi_k, stochrsi_d, reason,
                    extra={
                        'buy_price': removed.get('buy_price'),
                        'regime': removed.get('regime'),
                        'rsi_threshold_used': removed.get('rsi_threshold'),
                        'take_profit_pct_used': None,
                        'stop_loss_pct_used': None,
                        'trailing_tp_pct_used': removed.get('trailing_tp_pct'),
                        'trailing_sl_pct_used': removed.get('trailing_sl_pct'),
                        'buy_cooldown_used': None,
                        'position_size_used': None,
                    }
                )

        except Exception:
            logger.exception(f"[{symbol}] Error en monitoring_open_position")

        # 6) Dormir fuera del lock
        STOP_EVENT.wait(30)
# listen telegram messages
# ...

def listen_telegram_commands(token, chat_id, symbols, locks):
    last_update_id = None
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    logger.info("Escuchando comandos de Telegram...")

    if not token or not chat_id:
        logger.info("Telegram no configurado; listener no se inicia")
        return

    # 1) Drenar updates pendientes al iniciar
    last_update_id = init_telegram_offset(url, chat_id)

    while not STOP_EVENT.is_set():
        try:
            params = {"timeout": 60}
            if last_update_id is not None:
                params["offset"] = last_update_id + 1

            response = _http_request("GET", url, params=params, timeout=65, tries=3, log_prefix="Telegram getUpdates")
            data = response.json()

            if "result" in data:
                for update in data["result"]:
                    # 2) Ignorar updates viejos o de otros chats
                    if "message" not in update:
                        last_update_id = update["update_id"]
                        continue
                    msg = update["message"]
                    if str(msg.get("chat", {}).get("id")) != str(chat_id):
                        last_update_id = update["update_id"]
                        continue
                    if msg.get("date", 0) < STARTED_AT:
                        last_update_id = update["update_id"]
                        continue

                    text = (msg.get("text") or "").strip().lower()
                    last_update_id = update["update_id"]  # actualizar antes de ejecutar acción
                    logger.info(f"Comando recibido: {text}")

                    if text == "/posiciones":
                        for symbol in symbols:
                            # Pedir histórico suficiente para EMA200/ATR/ADX (en testnet puede ser limitado)
                            data_symbol = get_data_binance(symbol, interval=timeframe, limit=260)
                            send_positions_to_telegram(symbol, data_symbol, token, chat_id)

                    elif text == "/sellall":
                        for symbol in symbols:
                            with locks[symbol]:
                                positions = load_positions(symbol)
                                save_positions(symbol, [])
                            if not positions:
                                continue
                            price = get_precio_actual(symbol)
                            try:
                                symbol_filters = get_symbol_filters(symbol)
                            except Exception:
                                symbol_filters = None
                            for position in positions:
                                bal = get_base_asset_balance(symbol)
                                free_qty = float(bal.get('free', 0.0) or 0.0)
                                locked_qty = float(bal.get('locked', 0.0) or 0.0)

                                requested_qty = float(position.get('amount', 0.0) or 0.0)
                                sell_qty_raw = min(requested_qty, free_qty)
                                sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, price, for_sell=True, filters=symbol_filters)

                                if sell_qty is None and cancel_open_orders_on_stop and locked_qty > 0:
                                    canceled = cancel_all_open_orders(symbol)
                                    if canceled > 0:
                                        bal = get_base_asset_balance(symbol)
                                        free_qty = float(bal.get('free', 0.0) or 0.0)
                                        locked_qty = float(bal.get('locked', 0.0) or 0.0)
                                        sell_qty_raw = min(requested_qty, free_qty)
                                        sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, price, for_sell=True, filters=symbol_filters)

                                if sell_qty is None:
                                    min_qty_dbg = (symbol_filters or {}).get('min_qty')
                                    step_dbg = (symbol_filters or {}).get('step_size')
                                    logger.info(
                                        f"[{symbol}] Skip sellall requested={requested_qty} free={free_qty} locked={locked_qty} "
                                        f"raw={sell_qty_raw} min_qty={min_qty_dbg} step={step_dbg} motivo={motivo}"
                                    )
                                    continue
                                resp = ejecutar_orden_con_confirmacion('sell', symbol, sell_qty)
                                if resp:
                                    # registrar pnl realizado en risk manager
                                    try:
                                        realized = (float(price) - float(position.get('buy_price', 0.0) or 0.0)) * float(sell_qty)
                                        equity_est = float(get_balance())
                                        RISK_MANAGER.record_realized_pnl(symbol, realized, pd.Timestamp.now(), equity_total=equity_est)
                                    except Exception:
                                        pass

                                    log_trade(
                                        pd.Timestamp.now(), symbol, 'sell', price, sell_qty,
                                        (price - position['buy_price']) * sell_qty, 0, 0, 0, 0, 'SELL ALL',
                                        extra={
                                            'buy_price': position.get('buy_price'),
                                            'regime': position.get('regime'),
                                            'rsi_threshold_used': position.get('rsi_threshold'),
                                            'take_profit_pct_used': None,
                                            'stop_loss_pct_used': None,
                                            'trailing_tp_pct_used': position.get('trailing_tp_pct'),
                                            'trailing_sl_pct_used': position.get('trailing_sl_pct'),
                                            'buy_cooldown_used': None,
                                            'position_size_used': None,
                                        }
                                    )
                        send_event_to_telegram(f"🚨 Todas las posiciones vendidas.", token, chat_id)

                    elif text == "/stop":
                        send_event_to_telegram("🛑 Bot detenido por comando.", token, chat_id)
                        # ACK explícito del último update antes de salir
                        try:
                            _http_request("GET", url, params={"offset": last_update_id + 1, "timeout": 1}, timeout=2, tries=1,
                                          log_prefix="Telegram ACK")
                        except Exception as e:
                            logger.warning(f"ACK /stop falló: {e}")
                        STOP_EVENT.set()
                        break

                    elif text in ("/balance", "/saldo"):
                        try:
                            usdt_free = get_balance()
                        except Exception as e:
                            usdt_free = None
                            logger.error(f"Error obteniendo balance USDT: {e}")

                        total_positions_value = 0.0
                        details = ""
                        for symbol in symbols:
                            with locks[symbol]:
                                positions = load_positions(symbol)
                            if not positions:
                                continue
                            price = get_precio_actual(symbol)
                            if price is None:
                                continue
                            for p in positions:
                                pos_value = float(p.get('amount', 0)) * price
                                total_positions_value += pos_value
                                details += f"{symbol}: {p.get('amount',0):.6f} @ {price:.2f} = {pos_value:.2f} USDT\n"

                        msg_out = ("💵 Balance disponible (USDT): "
                                   f"{usdt_free:.2f} USDT\n" if usdt_free is not None else
                                   "💵 Balance USDT: error al obtener\n")
                        msg_out += f"📦 Valor estimado posiciones abiertas: {total_positions_value:.2f} USDT\n"
                        if details:
                            msg_out += "\nDetalles:\n" + details
                        send_event_to_telegram(msg_out, token, chat_id)

                    elif text == "/risk":
                        # Reporte de riesgo por símbolo (no altera trading, solo observabilidad)
                        try:
                            usdt_free = get_balance()
                        except Exception:
                            usdt_free = 0.0

                        positions_by_symbol: dict[str, list[dict]] = {}
                        prices_by_symbol: dict[str, float] = {}

                        for sym in symbols:
                            try:
                                with locks[sym]:
                                    positions_by_symbol[sym] = load_positions(sym)
                            except Exception:
                                positions_by_symbol[sym] = []

                        for sym in symbols:
                            try:
                                px = get_precio_actual(sym)
                                if px is not None:
                                    prices_by_symbol[sym] = float(px)
                            except Exception:
                                pass

                        # Equity total estimado: USDT libre + valor de todas las posiciones abiertas
                        equity_total = float(usdt_free)
                        for sym, pos in positions_by_symbol.items():
                            px = prices_by_symbol.get(sym)
                            if px is None or not pos:
                                continue
                            equity_total += _positions_current_value(pos, float(px))

                        lines = []
                        lines.append(f"🛡️ *RISK*\nEquity estimado: {equity_total:.2f} USDT\nUSDT libre: {float(usdt_free):.2f} USDT\n")

                        now_ts = time.time()
                        for sym in symbols:
                            pos = positions_by_symbol.get(sym, [])
                            px = prices_by_symbol.get(sym)

                            committed = _positions_cost_basis(pos)
                            current_value = _positions_current_value(pos, float(px)) if (px is not None) else 0.0
                            floating_pnl = current_value - committed

                            # refrescar métricas internas del RiskManager con datos actuales
                            try:
                                if px is not None:
                                    RISK_MANAGER.observe(sym, positions=pos, price=float(px), equity_total=float(equity_total), now_ts=now_ts)
                            except Exception:
                                pass

                            snap = RISK_MANAGER.snapshot(sym, now_ts=now_ts)
                            disabled_until = float(snap.get('disabled_until_ts') or 0.0)
                            paused_s = max(0, int(disabled_until - now_ts)) if disabled_until else 0

                            dd_cur = float(snap.get('current_drawdown') or 0.0) * 100
                            dd_max = float(snap.get('max_daily_drawdown') or 0.0) * 100
                            floating_loss = float(snap.get('floating_loss') or max(0.0, -floating_pnl))
                            daily_realized = float(snap.get('daily_realized_pnl') or 0.0)

                            cap_frac = (committed / equity_total) * 100 if equity_total > 0 else 0.0

                            header = f"*{sym}*"
                            if paused_s > 0:
                                header += f" (PAUSADO {paused_s//3600}h{(paused_s%3600)//60:02d}m)"

                            lines.append(
                                f"{header}\n"
                                f"Precio: {px:.4f}\n" if px is not None else f"{header}\nPrecio: N/A\n"
                            )
                            lines.append(
                                f"Posiciones: {len(pos)}\n"
                                f"Capital comprometido: {committed:.2f} USDT ({cap_frac:.2f}% equity)\n"
                                f"Valor actual: {current_value:.2f} USDT\n"
                                f"PnL flotante: {floating_pnl:+.2f} USDT | Pérdida flotante: {floating_loss:.2f} USDT\n"
                                f"Drawdown actual: {dd_cur:.2f}% | Máx diario: {dd_max:.2f}%\n"
                                f"PnL realizado hoy: {daily_realized:+.2f} USDT\n"
                            )

                        lines.append(
                            "\nReglas:\n"
                            f"- Bloqueo por DD símbolo: > {MAX_SYMBOL_DRAWDOWN_FRAC*100:.1f}%\n"
                            f"- Bloqueo por capital comprometido: > {MAX_CAPITAL_COMMITTED_FRAC*100:.1f}% equity\n"
                            f"- Pausa 24h si pérdida diaria realizada: > {DAILY_LOSS_LIMIT_FRAC*100:.1f}%\n"
                        )

                        send_event_to_telegram("\n".join(lines), token, chat_id)

                    elif text == "/params":
                        # Configuración activa (estrategia + riesgo). No altera ejecución.
                        try:
                            testnet_env = (os.getenv('BINANCE_TESTNET', 'true') or 'true').strip().lower()
                            use_testnet = testnet_env in ('1', 'true', 'yes', 'y', 'on')
                        except Exception:
                            use_testnet = True

                        lines = []
                        lines.append("⚙️ *PARAMS*")
                        lines.append(f"Testnet: {use_testnet}")
                        lines.append(f"Símbolos: {', '.join(symbols)}")
                        lines.append(f"TIMEFRAME: `{timeframe}`")
                        lines.append(f"POLL_INTERVAL_SECONDS: {poll_interval}")
                        lines.append("")

                        lines.append("*Entradas (calidad > cantidad)*")
                        lines.append("- BEAR: *NO abre nuevas posiciones* (hard filter)")
                        lines.append("- Condición: `price > EMA200` y `EMA50 > EMA200` y RSI en rango")
                        lines.append(f"- RSI_CONFIRM_MIN/MAX: {RSI_CONFIRM_MIN:.0f} - {RSI_CONFIRM_MAX:.0f}")
                        lines.append("")

                        lines.append("*ATR / niveles dinámicos*")
                        lines.append(f"- ATR_WINDOW: {ATR_WINDOW}")
                        bull = ATR_MULTIPLIERS.get('BULL')
                        lat = ATR_MULTIPLIERS.get('LATERAL')
                        lines.append(f"- BULL: TP={bull['tp']}x ATR | SL={bull['sl']}x ATR | trailingSL={bull['trailing_sl']}x ATR")
                        lines.append(f"- LATERAL: TP={lat['tp']}x ATR | SL={lat['sl']}x ATR | trailingSL={lat['trailing_sl']}x ATR")
                        lines.append("")

                        lines.append("*Tamaño / cooldown por régimen*")
                        lines.append(f"- BULL: POSITION_SIZE={float(BULL.get('POSITION_SIZE', 0.0) or 0.0)} | BUY_COOLDOWN={int(BULL.get('BUY_COOLDOWN', 0) or 0)}s")
                        lines.append(f"- LATERAL: POSITION_SIZE={float(LATERAL.get('POSITION_SIZE', 0.0) or 0.0)} | BUY_COOLDOWN={int(LATERAL.get('BUY_COOLDOWN', 0) or 0)}s")
                        lines.append(f"- BEAR: POSITION_SIZE={float(BEAR.get('POSITION_SIZE', 0.0) or 0.0)} | BUY_COOLDOWN={int(BEAR.get('BUY_COOLDOWN', 0) or 0)}s (sin compras)")
                        lines.append("")

                        lines.append("*DCA* (por defecto desactivado)")
                        lines.append(f"- ENABLE_DCA: {ENABLE_DCA} (si se activa, nunca corre en BEAR)")
                        lines.append("")

                        lines.append("*Risk manager por símbolo*")
                        lines.append(f"- MAX_SYMBOL_DRAWDOWN_FRAC: {MAX_SYMBOL_DRAWDOWN_FRAC*100:.1f}%")
                        lines.append(f"- MAX_CAPITAL_COMMITTED_FRAC: {MAX_CAPITAL_COMMITTED_FRAC*100:.1f}% del equity")
                        lines.append(f"- DAILY_LOSS_LIMIT_FRAC: {DAILY_LOSS_LIMIT_FRAC*100:.1f}% (pausa {SYMBOL_COOLDOWN_AFTER_DAILY_LOSS_SECONDS//3600}h)")
                        lines.append("")

                        lines.append("*Operativa* (seguridad)")
                        lines.append(f"- CANCEL_OPEN_ORDERS_ON_STOP: {cancel_open_orders_on_stop}")

                        send_event_to_telegram("\n".join(lines), token, chat_id)

                    elif text == "/help":
                        lines = []
                        lines.append("ℹ️ *HELP* — comandos disponibles")
                        lines.append("")
                        lines.append("- `/posiciones`: muestra posiciones por símbolo y métricas rápidas")
                        lines.append("- `/balance` o `/saldo`: USDT libre + valor estimado de posiciones")
                        lines.append("- `/risk`: reporte de riesgo (committed, PnL flotante, DD, pausas)")
                        lines.append("- `/params`: muestra configuración activa (entradas, ATR, risk limits)")
                        lines.append("- `/sellall`: vende todas las posiciones abiertas (según filtros/qty)")
                        lines.append("- `/stop`: detiene el bot de forma ordenada")
                        send_event_to_telegram("\n".join(lines), token, chat_id)

            time.sleep(1)

        except Exception as e:
            logger.error(f"Error escuchando comandos de Telegram: {e}")
            time.sleep(5)

    logger.info("Listener Telegram detenido")

def get_symbol_filters(symbol):
    info = _binance_call(client_trade.get_symbol_info, symbol)
    lot = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
    min_qty = float(lot['minQty'])
    step = float(lot['stepSize'])
    max_qty = float(lot['maxQty'])
    notional_filter = next((f for f in info['filters'] if f['filterType'] in ('MIN_NOTIONAL','NOTIONAL')), None)
    min_notional = float(notional_filter['minNotional']) if notional_filter else 0.0
    return {
        'min_qty': min_qty,
        'step_size': step,
        'max_qty': max_qty,
        'min_notional': min_notional
    }

def floor_to_step(qty: float, step: float) -> float:
    """
    Ajusta qty hacia abajo al múltiplo válido de step usando Decimal (sin errores binarios).
    Devuelve 0.0 si step es inválido o si qty < step.
    """
    try:
        d_qty = Decimal(str(qty))
        d_step = Decimal(str(step))
        if d_step <= 0:
            logger.error(f"floor_to_step: step inválido={step}")
            return 0.0
        steps = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN)
        adjusted = steps * d_step
        return float(adjusted)
    except Exception as e:
        logger.error(f"floor_to_step error qty={qty} step={step}: {e}")
        return 0.0


def sanitize_quantity(symbol: str, qty: float, price: float, for_sell: bool = False, filters: dict | None = None):
    """
    Ajusta qty al step y valida min_qty y min_notional.
    Devuelve (qty_ajustada | None, motivo | None)
    """
    try:
        f = filters or get_symbol_filters(symbol)
    except Exception as e:
        return None, f"no filters: {e}"
    adj = floor_to_step(qty, f['step_size'])
    if adj < f['min_qty']:
        return None, f"qty<{f['min_qty']}"
    if adj * price < f['min_notional']:
        return None, f"notional<{f['min_notional']}"
    return adj, None

def get_free_base_asset(symbol: str) -> float:
    info = _binance_call(client_trade.get_symbol_info, symbol)
    base = info['baseAsset']
    try:
        bal = _binance_call(client_trade.get_asset_balance, asset=base)
        if not bal:
            return 0.0
        return float(bal.get('free', 0.0))
    except Exception:
        return 0.0


def get_base_asset_balance(symbol: str) -> dict:
    """Retorna balance del activo base del símbolo: free/locked.

    Útil para diagnosticar casos donde la posición en CSV existe pero el balance libre es bajo
    porque está bloqueado en órdenes abiertas (locked) o ya no está disponible.
    """
    info = _binance_call(client_trade.get_symbol_info, symbol)
    base = info['baseAsset']
    try:
        bal = _binance_call(client_trade.get_asset_balance, asset=base)
        if not bal:
            return {'asset': base, 'free': 0.0, 'locked': 0.0}
        return {
            'asset': base,
            'free': float(bal.get('free', 0.0) or 0.0),
            'locked': float(bal.get('locked', 0.0) or 0.0),
        }
    except Exception:
        return {'asset': base, 'free': 0.0, 'locked': 0.0}


def cancel_all_open_orders(symbol: str) -> int:
    """Cancela órdenes abiertas del símbolo para liberar balance bloqueado."""
    try:
        orders = _binance_call(client_trade.get_open_orders, symbol=symbol, tries=2)
        if not orders:
            return 0
        canceled = 0
        for o in orders:
            oid = o.get('orderId')
            if oid is None:
                continue
            try:
                _binance_call(client_trade.cancel_order, symbol=symbol, orderId=oid, tries=2)
                canceled += 1
            except Exception as e:
                logger.warning(f"[{symbol}] No se pudo cancelar orderId={oid}: {e}")
        if canceled:
            logger.warning(f"[{symbol}] Canceladas {canceled} órdenes abiertas para liberar balance")
        return canceled
    except Exception as e:
        logger.warning(f"[{symbol}] Error obteniendo/cancelando órdenes abiertas: {e}")
        return 0

# --- Ajuste preparar_cantidad (reemplaza la versión actual) ---
def preparar_cantidad(symbol, usd_balance_frac, price, filters: dict | None = None):
    """
    Calcula una cantidad válida según filtros de Binance para el símbolo.
    Retorna (qty, motivo_error|None).
    """
    try:
        filters = filters or get_symbol_filters(symbol)
    except Exception as e:
        return None, f"no symbol filters ({e})"
    if not price or price <= 0:
        return None, f"invalid price ({price})"

    raw_qty = usd_balance_frac / price
    step = filters['step_size']

    # Ajustar con Decimal para evitar errores binarios y rechazos por LOT_SIZE.
    qty = floor_to_step(raw_qty, step)

    # Respetar max_qty si existe
    try:
        max_qty = float(filters.get('max_qty', 0) or 0)
        if max_qty > 0 and qty > max_qty:
            qty = floor_to_step(max_qty, step)
    except Exception:
        pass

    qty = float(f"{qty:.15f}")  # normalizar
    if qty < filters['min_qty']:
        return None, f"qty < min_qty ({qty} < {filters['min_qty']})"
    notional = qty * price
    if notional < filters['min_notional']:
        return None, f"notional < min_notional ({notional} < {filters['min_notional']})"
    return qty, None

# --- Agregar helper de debug de filtros (opcional para investigar problemas) ---
def debug_symbol_filters(symbol):
    f = get_symbol_filters(symbol)
    logger.debug(f"Filtros {symbol}: min_qty={f['min_qty']} step={f['step_size']} max_qty={f['max_qty']} min_notional={f['min_notional']}")

def detect_market_regime(df):
    """Detecta el régimen del mercado (BULL/BEAR/LATERAL) usando EMA200 y ADX.

    Si no hay datos suficientes o hay NaN, retorna LATERAL.
    """
    try:
        if df is None or df.empty:
            return "LATERAL"
        if 'close' not in df.columns or 'ema200' not in df.columns or 'adx' not in df.columns:
            return "LATERAL"

        price = df['close'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        adx = df['adx'].iloc[-1]

        if pd.isna(price) or pd.isna(ema200) or pd.isna(adx):
            return "LATERAL"

        if price > ema200 and adx >= ADX_REGIME_MIN:
            return "BULL"
        elif price < ema200 and adx >= ADX_REGIME_MIN:
            return "BEAR"
        else:
            return "LATERAL"
    except Exception:
        return "LATERAL"


# Parámetros de estrategia por régimen de mercado.
# Nota: Se aplican SOLO a operaciones nuevas. Las posiciones ya abiertas mantienen sus
# niveles (take_profit/stop_loss) y, si no tienen trailing por posición, usan el trailing global.
BULL = {
    "BUY_COOLDOWN": 7200,      # 2h
    "POSITION_SIZE": 0.08
}

BEAR = {
    "BUY_COOLDOWN": 21600,     # 6h
    "POSITION_SIZE": 0.0
}

LATERAL = {
    "BUY_COOLDOWN": 14400,     # 4h
    "POSITION_SIZE": 0.03
}

REGIME_PARAMS = {
    "BULL": BULL,
    "BEAR": BEAR,
    "LATERAL": LATERAL,
}


# Risk manager global (no toca red, thread-safe)
RISK_MANAGER = RiskManager(SYMBOLS)

def _build_v5_strategy_adapter():
    """Construye el adaptador de estrategia v5 leyendo env vars.

    Lee V5_MODEL_PATH (override de ML_MODEL_PATH) y V5_THRESHOLD_OVERRIDE.
    Usado en run_strategy y reemplazable en tests sin tocar el módulo ML.
    """
    from estrategia_v5 import BotV5StrategyAdapter
    model_path = (os.getenv('V5_MODEL_PATH') or '').strip() or ML_MODEL_PATH
    thr_str = (os.getenv('V5_THRESHOLD_OVERRIDE') or '').strip()
    threshold_override = float(thr_str) if thr_str else None
    return BotV5StrategyAdapter(
        model_path=model_path,
        threshold_override=threshold_override,
    )


def _build_strategy_engine() -> tuple[StrategyEngine, PortfolioManager]:
    """Construye el StrategyEngine y PortfolioManager del sistema multi-estrategia v2.

    Inyectable en tests: reemplazar _build_strategy_engine para evitar carga del modelo ML.
    """
    from strategy_engine import build_default_engine as _build_engine
    engine = _build_engine()
    pm = PortfolioManager(strategies=engine.strategies)
    return engine, pm


# Ejecutar la estrategia en tiempo real
def run_strategy(symbol, lock):
    logger.info(f"[{symbol}] Iniciando hilo de estrategia")
    _require_clients()

    # Construir adaptador de estrategia (inyectable en tests via _build_v5_strategy_adapter mock)
    strategy = _build_v5_strategy_adapter()

    balance = get_balance()
    logger.info(f"[{symbol}] Balance inicial: {balance:.2f} USDT")
    last_buy_time: float | None = None
    positions_cache = []

    # Cachear filtros por símbolo para evitar llamadas repetidas
    try:
        symbol_filters = get_symbol_filters(symbol)
    except Exception as e:
        logger.warning(f"[{symbol}] No se pudieron obtener filtros de símbolo: {e}")
        symbol_filters = None

    # Constantes de estrategia: fuera del loop para no recrearlas en cada iteración.
    # Señales siempre sobre vela CERRADA (iloc[-2]) para evitar lookahead.
    POSITION_SIZE_SIMPLE = 0.03       # 3% del balance disponible
    trailing_sl_atr_mult_fixed = 1.2  # multiplicador ATR para trailing stop

    while not STOP_EVENT.is_set():
        try:
            # Intento no bloqueante sobre el lock (máx 1s)
            positions = None
            acquired = lock.acquire(timeout=1.0)
            if acquired:
                try:
                    positions = load_positions(symbol)
                    positions_cache = positions  # actualizar cache
                finally:
                    lock.release()
            else:
                positions = positions_cache  # usar último snapshot

            data = get_data_binance(symbol, interval=timeframe, limit=260)
            if data is None or data.empty:
                logger.warning(f"[{symbol}] Sin datos de mercado en DB. Omitiendo ciclo.")
                STOP_EVENT.wait(poll_interval)
                continue
            # Cross-asset features (eth_correlation_30, btc_dominance): necesitan DB/red.
            # Deben estar en el df ANTES de que prepare_indicators llame a _ensure_ml_features.
            try:
                b = _get_ml_bundle()
                if b and b.get('features'):
                    feat_list = list(b.get('features') or [])
                    data = _add_cross_asset_features(data, symbol=symbol, interval=timeframe, features=feat_list)
            except Exception:
                pass

            # prepare_indicators: calcula todos los indicadores técnicos y ML features locales.
            df = strategy.prepare_indicators(symbol=symbol, df=data)

            row = df.iloc[-2]
            close_price = row['close']
            high = row['high']
            low = row['low']
            open_price = row['open']
            timestamp = row.get('timestamp', pd.Timestamp.now())

            ema200 = row['ema200']
            ema50 = row['ema50']
            adx = row['adx']
            atr = row['atr']
            volatility = (float(high) - float(low)) / float(open_price) * 100 if open_price else 0

            # Guardas: si indicadores aún no están disponibles, no operar
            if pd.isna(ema200) or pd.isna(ema50) or pd.isna(adx) or pd.isna(atr) or pd.isna(close_price):
                logger.warning(f"[{symbol}] Indicadores no disponibles (NaN). Omitiendo ciclo.")
                STOP_EVENT.wait(poll_interval)
                continue

            try:
                atr = float(atr)
            except Exception:
                STOP_EVENT.wait(poll_interval)
                continue
            if atr <= 0:
                logger.warning(f"[{symbol}] ATR inválido ({atr}). Omitiendo ciclo.")
                STOP_EVENT.wait(poll_interval)
                continue

            try:
                close_f = float(close_price)
                ema200_f = float(ema200)
                ema50_f = float(ema50)
                adx_f = float(adx)
            except Exception:
                STOP_EVENT.wait(poll_interval)
                continue

            # Régimen calculado por prepare_indicators (sin lookahead, misma lógica que backtesting)
            regime = str(row.get('regime', 'LATERAL')).upper()

            logger.info(
                f"[{symbol}] Precio: {close_f:.4f} | EMA200: {ema200_f:.4f} | EMA50: {ema50_f:.4f} | "
                f"ADX: {adx_f:.2f} | ATR: {float(atr):.4f} | Regime: {regime} | Posiciones: {len(positions)}"
            )

            # Refrescar balance solo cuando sea relevante (evita llamadas constantes)
            balance = get_balance()

            # Preferir min_notional real del símbolo si está disponible
            symbol_min_notional = (symbol_filters or {}).get('min_notional', min_notional)

            # Alimentar risk manager (métricas intradía) con equity estimado
            equity_est = float(balance)
            try:
                equity_est = float(balance) + _positions_current_value(positions, float(close_price))
                RISK_MANAGER.observe(symbol, positions=positions, price=float(close_price), equity_total=float(equity_est))
            except Exception:
                pass

            # Cooldown: evita reentradas inmediatas tras una compra.
            # BUY_COOLDOWN_SECONDS controla el tiempo mínimo entre compras del mismo símbolo.
            if last_buy_time is not None:
                elapsed = time.time() - last_buy_time
                if elapsed < cooldown_seconds:
                    remaining = int(cooldown_seconds - elapsed)
                    _log_throttled(
                        f'cooldown_{symbol}', logging.INFO,
                        '[%s] Cooldown activo (%ss restantes). Omitiendo entrada.',
                        symbol, remaining,
                        every_s=max(60.0, cooldown_seconds / 4),
                    )
                    STOP_EVENT.wait(poll_interval)
                    continue

            # --- Sistema multi-estrategia v2: StrategyEngine + PortfolioManager ---
            # Construye MarketState para pasar a todas las estrategias.
            from strategies.base_strategy import MarketState as _MarketState
            _market_state = _MarketState(
                symbol=symbol,
                df=df,
                regime=regime,
                balance=float(balance),
                equity=float(equity_est),
                open_positions=list(positions),
                indicators=dict(row),
            )

            # Evaluar todas las estrategias elegibles (fail-safe por estrategia)
            _portfolio_order: PortfolioOrder | None = None
            try:
                if STRATEGY_ENGINE is not None and PORTFOLIO_MANAGER is not None:
                    _signals = STRATEGY_ENGINE.collect(_market_state)
                    _portfolio_order = PORTFOLIO_MANAGER.decide(symbol, _signals)
                    if not _portfolio_order.should_enter:
                        logger.debug(
                            "[%s] PortfolioManager: HOLD score=%.3f triggered=%s vetoed=%s",
                            symbol,
                            _portfolio_order.score,
                            _portfolio_order.triggered_by,
                            _portfolio_order.vetoed_by,
                        )
            except Exception as _pe:
                logger.warning("[%s] StrategyEngine/PortfolioManager falló (fail-open): %s", symbol, _pe)
                _portfolio_order = None

            # Fallback: señal ML directa (compatibilidad con estrategia única)
            ctx = StrategyContext(
                symbol=symbol,
                i=len(df) - 2,
                timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(tz=timezone.utc),
                indicators=dict(row),
                regime=regime,
                cash=float(balance),
                equity=float(equity_est),
                positions_open_symbol=len(positions),
                last_entry_time=last_buy_time,
            )
            sig = strategy.generate_entry(ctx)

            # Determinar si entrar y el tamaño de posición
            if _portfolio_order is not None:
                entry_ok = _portfolio_order.should_enter
                position_size_active = (
                    float(_portfolio_order.size_frac) if _portfolio_order.size_frac > 0
                    else float(POSITION_SIZE_SIMPLE)
                )
                _entry_meta = _portfolio_order.meta
            else:
                # Fallback: solo señal ML (STRATEGY_ENGINE no disponible)
                entry_ok = sig.should_enter
                position_size_active = (
                    float(sig.position_size_frac) if (sig.position_size_frac is not None and float(sig.position_size_frac) > 0)
                    else float(POSITION_SIZE_SIMPLE)
                )
                _entry_meta = sig.meta or {}

            # Filtro adicional: pendiente EMA200 debe ser positiva para confirmar tendencia alcista.
            # Bloquea entradas en rallies dentro de mercados bajistas (EMA200 declinante).
            if entry_ok and EMA200_SLOPE_FILTER:
                try:
                    ema200_slope_val = row.get("ema200_slope")
                    if ema200_slope_val is not None and not pd.isna(ema200_slope_val):
                        if float(ema200_slope_val) <= 0:
                            logger.info(
                                "[%s] EMA200 slope negativa (%.4f) — entrada bloqueada (filtro tendencia alcista).",
                                symbol, float(ema200_slope_val),
                            )
                            entry_ok = False
                except Exception:
                    pass

            if entry_ok:
                logger.info(
                    "[%s] Señal entrada: regime=%s ml_prob=%s close=%.4f ema50=%.4f ema200=%.4f adx=%.2f triggered=%s",
                    symbol,
                    regime,
                    (_entry_meta.get('ml_prob') if _portfolio_order is None else
                     _entry_meta.get('strategies_buy', [])),
                    close_f,
                    ema50_f,
                    ema200_f,
                    adx_f,
                    _portfolio_order.triggered_by if _portfolio_order else [],
                )

            # Intento compra principal
            executed = False

            # Gestión de posiciones: máximo 1 posición por símbolo
            if entry_ok and balance > symbol_min_notional and len(positions) == 0:

                # Risk manager global (por símbolo) antes de abrir
                symbol_committed = _positions_cost_basis(positions)

                # Equity total real (USDT libre + valor de posiciones en todos los símbolos)
                equity_total = compute_equity_total(SYMBOLS, None, price_overrides={symbol: float(close_price)})
                # drawdown actual basado en peak intradía
                try:
                    sym_value = _positions_current_value(positions, float(close_price))
                    snap = RISK_MANAGER.snapshot(symbol)
                    peak = float(snap.get('peak_value') or 0.0)
                    symbol_drawdown = max(0.0, (peak - sym_value) / peak) if peak > 0 else 0.0
                except Exception:
                    symbol_drawdown = 0.0

                allowed, reason = RISK_MANAGER.can_open(
                    symbol,
                    equity_total=float(equity_total),
                    committed_capital=float(symbol_committed),
                    symbol_drawdown=float(symbol_drawdown),
                )
                if not allowed:
                    logger.warning(f"[{symbol}] RiskManager bloquea nueva entrada: {reason}")
                    STOP_EVENT.wait(poll_interval)
                    continue

                capital_usar = balance * position_size_active

                # Risk Layer v3 (opcional, fail-open): sizing por ATR
                qty_candidate = None
                try:
                    if V3_POSITION_SIZER is not None and bool(getattr(V3_POSITION_SIZER, 'enabled', False)):
                        qty_candidate = float(
                            V3_POSITION_SIZER.compute_position_size(
                                symbol=symbol,
                                equity_total=float(equity_total),
                                atr=float(atr),
                                price=float(close_price),
                            )
                        )
                        if qty_candidate > 0:
                            capital_usar = float(qty_candidate) * float(close_price)
                except Exception as e:
                    logger.warning(f"[{symbol}] Risk v3 sizing falló (fail-open): {e}")

                # Equity regime (block mode) antes de abrir
                try:
                    if V3_EQUITY_REGIME_FILTER is not None:
                        allowed_v3_regime, reason_v3_regime = V3_EQUITY_REGIME_FILTER.can_open()
                        if not allowed_v3_regime:
                            logger.warning(f"[{symbol}] RISK_V3_BLOCK: {reason_v3_regime}")
                            STOP_EVENT.wait(poll_interval)
                            continue
                except Exception as e:
                    logger.warning(f"[{symbol}] Risk v3 equity regime can_open falló (fail-open): {e}")

                # Correlación de portfolio con exposición combinada (incluye propuesta)
                try:
                    if V3_CORRELATION_RISK is not None:
                        all_positions = load_all_open_positions()
                        proposal_qty = float(capital_usar) / float(close_price) if float(close_price) > 0 else 0.0
                        proposal = {
                            'symbol': symbol,
                            'amount': float(max(0.0, proposal_qty)),
                            'current_price': float(close_price),
                        }
                        allowed_v3_corr, reason_v3_corr = V3_CORRELATION_RISK.can_open(
                            symbol,
                            current_positions=[*all_positions, proposal],
                            equity_total=float(equity_total),
                        )
                        if not allowed_v3_corr:
                            logger.warning(f"[{symbol}] RISK_V3_BLOCK: {reason_v3_corr}")
                            STOP_EVENT.wait(poll_interval)
                            continue
                except Exception as e:
                    logger.warning(f"[{symbol}] Risk v3 correlation falló (fail-open): {e}")

                # VaR intradía
                try:
                    if V3_VAR_MONITOR is not None:
                        allowed_v3_var, reason_v3_var = V3_VAR_MONITOR.can_open(float(equity_total))
                        if not allowed_v3_var:
                            logger.warning(f"[{symbol}] RISK_V3_BLOCK: {reason_v3_var}")
                            STOP_EVENT.wait(poll_interval)
                            continue
                except Exception as e:
                    logger.warning(f"[{symbol}] Risk v3 VaR falló (fail-open): {e}")

                # Ajuste de tamaño por equity regime (reduce mode)
                try:
                    if V3_EQUITY_REGIME_FILTER is not None:
                        capital_usar = float(V3_EQUITY_REGIME_FILTER.adjust_position_size(float(capital_usar)))
                except Exception as e:
                    logger.warning(f"[{symbol}] Risk v3 equity regime adjust falló (fail-open): {e}")

                qty, motivo = preparar_cantidad(symbol, capital_usar, close_price, filters=symbol_filters)
                if qty is None:
                    logger.info(f"[{symbol}] Skip compra (condición principal): {motivo}")
                else:
                    debug_symbol_filters(symbol)

                    entry_price = float(close_f)
                    take_profit, stop_loss, risk_extra = strategy.compute_risk_levels(
                        symbol=symbol,
                        regime=regime,
                        buy_price=entry_price,
                        indicators_row=dict(row),
                    )

                    resp = ejecutar_orden_con_confirmacion('buy', symbol, qty)
                    if resp:
                        try:
                            if V3_SLIPPAGE_MONITOR is not None:
                                executed_qty = float(resp.get('executedQty', 0) or 0)
                                quote_qty = float(resp.get('cummulativeQuoteQty', 0) or 0)
                                if executed_qty > 0 and quote_qty > 0:
                                    fill_px = quote_qty / executed_qty
                                    V3_SLIPPAGE_MONITOR.record_fill(
                                        expected_price=float(entry_price),
                                        fill_price=float(fill_px),
                                    )
                        except Exception as e:
                            logger.warning(f"[{symbol}] Risk v3 slippage falló (fail-open): {e}")

                        _sig_meta = _entry_meta
                        new_position = {
                            'buy_price': float(entry_price),
                            'amount': qty,
                            'timestamp': timestamp,
                            'regime': str(regime),
                            'take_profit': float(take_profit),
                            'stop_loss': float(stop_loss),
                            'tp_initial': float(risk_extra.get('tp_initial', take_profit)),
                            'atr_entry': float(risk_extra.get('atr_entry', atr)),
                            'tp_atr_mult': float(risk_extra.get('tp_atr_mult', 2.5)),
                            'sl_atr_mult': float(risk_extra.get('sl_atr_mult', 1.5)),
                            'trailing_sl_atr_mult': float(risk_extra.get('trailing_sl_atr_mult', trailing_sl_atr_mult_fixed)),
                            'trailing_active': bool(risk_extra.get('trailing_active', False)),
                            'max_price': float(risk_extra.get('max_price', entry_price)),
                            'ml_prob': _sig_meta.get('ml_prob'),
                            'ml_threshold': _sig_meta.get('ml_threshold'),
                        }
                        with lock:
                            current = load_positions(symbol)
                            current.append(new_position)
                            save_positions(symbol, current)
                        balance = get_balance()
                        last_buy_time = time.time()
                        send_event_to_telegram(f'📈 COMPRA {symbol}: {qty:.6f} @ {close_price:.4f}', TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                        log_trade(
                            timestamp, symbol, 'buy', close_price, qty, 0,
                            volatility,
                            (df['rsi'].iloc[-2] if 'rsi' in df.columns else None),
                            (df['stochrsi_k'].iloc[-2] if 'stochrsi_k' in df.columns else None),
                            (df['stochrsi_d'].iloc[-2] if 'stochrsi_d' in df.columns else None),
                            'BUY',
                            extra={
                                # Estrategia simple (sin régimen)
                                'take_profit_pct_used': None,
                                'stop_loss_pct_used': None,
                                'trailing_tp_pct_used': None,
                                'trailing_sl_pct_used': float(trailing_sl_atr_mult_fixed),
                                'buy_cooldown_used': None,
                                'position_size_used': position_size_active,
                                'regime': str(regime),
                                'ml_prob': _sig_meta.get('ml_prob'),
                                'ml_threshold': _sig_meta.get('ml_threshold'),
                            }
                        )
                        executed = True

            # DCA: compra adicional cuando el mercado cae >= 5% en las últimas 5 velas.
            # Solo activo si ENABLE_DCA=true en .env. Nunca en BEAR.
            # Usa 50% del position_size normal para no sobreexponer el capital.
            if ENABLE_DCA and (not executed) and regime != "BEAR" and len(positions) == 0:
                try:
                    movimiento = market_change_last_5_intervals(symbol)
                    if movimiento is not None and balance > symbol_min_notional and movimiento <= -5:
                        logger.info(
                            f"[{symbol}] DCA: condición activada movimiento={movimiento:.2f}% — "
                            f"intentando compra reducida (50% position_size)"
                        )

                        # Tamaño reducido al 50% para DCA — evitar martingala agresivo
                        capital_dca = balance * (position_size_active * 0.5)

                        qty_dca, motivo_dca = preparar_cantidad(
                            symbol, capital_dca, close_price, filters=symbol_filters
                        )

                        if qty_dca is None:
                            logger.info(f"[{symbol}] DCA skip: {motivo_dca}")
                        else:
                            entry_price_dca = float(close_f)
                            take_profit_dca, stop_loss_dca, risk_extra_dca = strategy.compute_risk_levels(
                                symbol=symbol,
                                regime=regime,
                                buy_price=entry_price_dca,
                                indicators_row=dict(row),
                            )

                            resp_dca = ejecutar_orden_con_confirmacion('buy', symbol, qty_dca)
                            if resp_dca:
                                # Registrar slippage (Risk v3)
                                try:
                                    if V3_SLIPPAGE_MONITOR is not None:
                                        exec_qty = float(resp_dca.get('executedQty', 0) or 0)
                                        quote_qty = float(resp_dca.get('cummulativeQuoteQty', 0) or 0)
                                        if exec_qty > 0 and quote_qty > 0:
                                            V3_SLIPPAGE_MONITOR.record_fill(
                                                expected_price=float(entry_price_dca),
                                                fill_price=float(quote_qty / exec_qty),
                                            )
                                except Exception as e:
                                    logger.warning(f"[{symbol}] DCA slippage monitor falló: {e}")

                                new_position_dca = {
                                    'buy_price':            float(entry_price_dca),
                                    'amount':               qty_dca,
                                    'timestamp':            timestamp,
                                    'regime':               str(regime),
                                    'take_profit':          float(take_profit_dca),
                                    'stop_loss':            float(stop_loss_dca),
                                    'tp_initial':           float(risk_extra_dca.get('tp_initial', take_profit_dca)),
                                    'atr_entry':            float(risk_extra_dca.get('atr_entry', atr)),
                                    'tp_atr_mult':          float(risk_extra_dca.get('tp_atr_mult', 2.5)),
                                    'sl_atr_mult':          float(risk_extra_dca.get('sl_atr_mult', 1.5)),
                                    'trailing_sl_atr_mult': float(risk_extra_dca.get('trailing_sl_atr_mult', trailing_sl_atr_mult_fixed)),
                                    'trailing_active':      bool(risk_extra_dca.get('trailing_active', False)),
                                    'max_price':            float(risk_extra_dca.get('max_price', entry_price_dca)),
                                    'ml_prob':              None,   # DCA no usa filtro ML
                                    'ml_threshold':         None,
                                    'dca':                  True,   # Marca la posición como DCA
                                    'dca_movimiento':       float(movimiento),
                                }
                                with lock:
                                    current = load_positions(symbol)
                                    current.append(new_position_dca)
                                    save_positions(symbol, current)

                                balance = get_balance()
                                last_buy_time = time.time()

                                send_event_to_telegram(
                                    f'📉 DCA {symbol}: {qty_dca:.6f} @ {close_price:.4f} '
                                    f'(caída {movimiento:.2f}%)',
                                    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                )
                                log_trade(
                                    timestamp, symbol, 'buy', close_price, qty_dca, 0,
                                    volatility,
                                    (df['rsi'].iloc[-2] if 'rsi' in df.columns else None),
                                    (df['stochrsi_k'].iloc[-2] if 'stochrsi_k' in df.columns else None),
                                    (df['stochrsi_d'].iloc[-2] if 'stochrsi_d' in df.columns else None),
                                    'DCA',
                                    extra={
                                        'take_profit_pct_used':  None,
                                        'stop_loss_pct_used':    None,
                                        'trailing_tp_pct_used':  None,
                                        'trailing_sl_pct_used':  float(trailing_sl_atr_mult_fixed),
                                        'buy_cooldown_used':     None,
                                        'position_size_used':    position_size_active * 0.5,
                                        'regime':                str(regime),
                                        'dca_movimiento':        float(movimiento),
                                        'ml_prob':               None,
                                        'ml_threshold':          None,
                                    }
                                )
                                executed = True
                            else:
                                logger.error(f"[{symbol}] DCA orden fallida qty={qty_dca}")
                except Exception as e:
                    logger.warning(f"[{symbol}] DCA bloque falló (continuando): {e}")
            
            STOP_EVENT.wait(poll_interval)

        except Exception as e:
            logger.exception(f"[{symbol}] Error en run_strategy")
            STOP_EVENT.wait(5)

def main():
    global client_data, client_trade
    global DB, OPEN_POS_REPO, TRADE_REPO, EQUITY_REPO
    global RISK_EVENT_LOGGER, GLOBAL_RISK_CONTROLLER, SYSTEM_HEALTH_MONITOR
    global V3_POSITION_SIZER, V3_CORRELATION_RISK, V3_VAR_MONITOR, V3_SLIPPAGE_MONITOR, V3_EQUITY_REGIME_FILTER

    # Validaciones y configuración
    _validate_required_env()

    use_db_flag = str(os.getenv('USE_DATABASE', '') or '').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    if not use_db_flag:
        raise RuntimeError("USE_DATABASE=true es obligatorio. CSV/file persistence fue deshabilitado.")

    # En esta arquitectura:
    # - MAINNET: solo datos de mercado (klines/ticker)
    # - TESTNET: solo ejecución (órdenes/balances/cancel)
    # Nunca ejecutar órdenes en MAINNET.

    # Señales para detener ordenadamente
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    # Inicializar clientes Binance
    trade_key = (os.getenv('BINANCE_TRADE_API_KEY') or os.getenv('BINANCE_API_KEY') or '').strip() or None
    trade_secret = (os.getenv('BINANCE_TRADE_API_SECRET') or os.getenv('BINANCE_API_SECRET') or '').strip() or None
    if not trade_key or not trade_secret:
        raise RuntimeError(
            "Faltan credenciales de trading TESTNET. Configura BINANCE_TRADE_API_KEY y BINANCE_TRADE_API_SECRET "
            "(o usa BINANCE_API_KEY/BINANCE_API_SECRET como fallback)."
        )

    data_key = (os.getenv('BINANCE_DATA_API_KEY') or '').strip() or None
    data_secret = (os.getenv('BINANCE_DATA_API_SECRET') or '').strip() or None

    client_data = Client(data_key, data_secret, testnet=False)
    client_trade = Client(trade_key, trade_secret, testnet=True)
    logger.info("Clientes Binance inicializados: data=MAINNET trade=TESTNET")

    # Validación temprana (falla rápido)
    try:
        _ = _binance_call(client_data.get_server_time, tries=1)
        logger.info("MAINNET data OK: server time verificado.")
    except Exception as e:
        logger.error("MAINNET data client falló en get_server_time: %s", e)
        raise

    try:
        _binance_call(client_trade.get_account, tries=1)
        logger.info("TESTNET trade OK: acceso a cuenta verificado.")
    except BinanceAPIException as be:
        code = getattr(be, 'code', None)
        msg = getattr(be, 'message', str(be))
        if str(code) == '-2015':
            logger.error(
                "Binance TESTNET auth falló (-2015): %s. Revisa: (1) API keys de TESTNET, "
                "(2) permisos habilitados (Read/Spot Trading), (3) restricción de IP/whitelist, "
                "(4) API key/secret sin espacios ni comillas.",
                msg,
            )
        raise

    # --- Inicializar PostgreSQL (OBLIGATORIO) ---
    # Mantiene el proceso vivo y reintenta hasta conectar (no inicia trading sin DB).
    while not STOP_EVENT.is_set():
        try:
            DB = init_db_from_env()
            ensure_schema(DB)
            OPEN_POS_REPO = OpenPositionsRepository(DB)
            TRADE_REPO = TradeHistoryRepository(DB)
            EQUITY_REPO = EquitySnapshotsRepository(DB)
            logger.info("PostgreSQL listo: storage habilitado (single source of truth).")
            break
        except Exception as e:
            logger.error(f"No se pudo inicializar PostgreSQL: {e}. Reintentando en 10s...")
            STOP_EVENT.wait(10)

    if STOP_EVENT.is_set():
        logger.warning("Detenido antes de iniciar trading (señal recibida durante init DB).")
        return

    # --- Risk Layer v2: inicialización (después de DB ready, antes de arrancar hilos) ---
    RISK_EVENT_LOGGER = RiskEventLogger(DB)

    def _send_telegram_msg(msg: str) -> None:
        try:
            send_event_to_telegram(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        except Exception:
            pass

    SYSTEM_HEALTH_MONITOR = SystemHealthMonitor(
        stop_event=STOP_EVENT,
        send_telegram=_send_telegram_msg,
        event_logger=RISK_EVENT_LOGGER,
    )
    GLOBAL_RISK_CONTROLLER = GlobalRiskController(
        db=DB,
        stop_event=STOP_EVENT,
        send_telegram=_send_telegram_msg,
        event_logger=RISK_EVENT_LOGGER,
    )

    # --- Risk Layer v3: opcional, modular y fail-open ---
    try:
        V3_POSITION_SIZER = VolatilityPositionSizer()
        V3_CORRELATION_RISK = PortfolioCorrelationRisk(db=DB, symbols=SYMBOLS)
        V3_VAR_MONITOR = IntradayVaRMonitor(db=DB)
        V3_SLIPPAGE_MONITOR = SlippageMonitor(
            stop_event=STOP_EVENT,
            event_logger=RISK_EVENT_LOGGER,
            send_telegram=_send_telegram_msg,
        )
        V3_EQUITY_REGIME_FILTER = EquityRegimeFilter(db=DB)

        logger.info(
            "Risk v3 init: sizer=%s corr=%s var=%s slippage=%s equity_regime=%s",
            bool(getattr(V3_POSITION_SIZER, 'enabled', False)),
            bool(getattr(V3_CORRELATION_RISK, 'enabled', False)),
            bool(getattr(V3_VAR_MONITOR, 'enabled', False)),
            bool(getattr(V3_SLIPPAGE_MONITOR, 'enabled', False)),
            bool(getattr(V3_EQUITY_REGIME_FILTER, 'enabled', False)),
        )
    except Exception as e:
        logger.warning("Risk v3 init falló (continuando sin v3): %s", e)
        V3_POSITION_SIZER = None
        V3_CORRELATION_RISK = None
        V3_VAR_MONITOR = None
        V3_SLIPPAGE_MONITOR = None
        V3_EQUITY_REGIME_FILTER = None

    # --- Motor multi-estrategia legacy: inicialización (fail-open) ---
    global MULTI_ENGINE
    try:
        MULTI_ENGINE = build_default_engine()
        logger.info("MultiStrategyEngine inicializado: mode=%s strategies=%s",
                    MULTI_ENGINE.mode,
                    [s.name for s in MULTI_ENGINE.strategies])
    except Exception as e:
        logger.warning("MultiStrategyEngine init falló (continuando sin multi-engine): %s", e)
        MULTI_ENGINE = None

    # --- Arquitectura multi-estrategia v2: StrategyEngine + PortfolioManager ---
    global STRATEGY_ENGINE, PORTFOLIO_MANAGER
    try:
        STRATEGY_ENGINE, PORTFOLIO_MANAGER = _build_strategy_engine()
        logger.info(
            "StrategyEngine inicializado: %d estrategias — %s",
            len(STRATEGY_ENGINE.strategies),
            [s.strategy_id for s in STRATEGY_ENGINE.strategies],
        )
        if PORTFOLIO_MANAGER.ml_hybrid_mode:
            logger.info(
                "PortfolioManager: buy_thr=%.2f veto_on_hold=%s "
                "modo=HÍBRIDO ml_min_conf=%.2f scales=(%.1fx/%.1fx/%.1fx)",
                PORTFOLIO_MANAGER.buy_threshold,
                PORTFOLIO_MANAGER.veto_on_hold,
                PORTFOLIO_MANAGER.ml_min_confidence,
                PORTFOLIO_MANAGER.ml_size_scale_low,
                PORTFOLIO_MANAGER.ml_size_scale_mid,
                PORTFOLIO_MANAGER.ml_size_scale_high,
            )
        else:
            logger.info(
                "PortfolioManager: buy_thr=%.2f sell_thr=%.2f veto_on_hold=%s modo=ESTÁNDAR",
                PORTFOLIO_MANAGER.buy_threshold,
                PORTFOLIO_MANAGER.sell_threshold,
                PORTFOLIO_MANAGER.veto_on_hold,
            )
    except Exception as e:
        logger.warning("StrategyEngine/PortfolioManager init falló (fail-open): %s", e)
        STRATEGY_ENGINE = None
        PORTFOLIO_MANAGER = None

    locks = {symbol: threading.Lock() for symbol in SYMBOLS}
    logger.info(f"Iniciando bot con símbolos: {SYMBOLS}")

    # +3: reconcile worker, risk metrics worker, health server
    with ThreadPoolExecutor(max_workers=(len(SYMBOLS) * 2) + 5) as executor:
        for symbol in SYMBOLS:
            executor.submit(monitoring_open_position, symbol, locks[symbol])
            executor.submit(run_strategy, symbol, locks[symbol])
        executor.submit(listen_telegram_commands, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, locks)
        executor.submit(equity_snapshot_worker, SYMBOLS, locks)

        # --- Reconciliación automática DB <-> Binance ---
        def _reconcile_one(sym: str) -> None:
            try:
                _require_repos()
                reconcile_positions_with_exchange(
                    symbol=sym,
                    db=DB,
                    open_pos_repo=OPEN_POS_REPO,  # type: ignore[arg-type]
                    client_trade=client_trade,
                    get_live_price=get_precio_actual,
                    event_logger=RISK_EVENT_LOGGER,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception("reconcile_one failed sym=%s", sym)

        # Run once at startup (non-fatal)
        for sym in SYMBOLS:
            _reconcile_one(sym)

        executor.submit(
            reconcile_worker,
            symbols=SYMBOLS,
            interval_s=600.0,
            stop_event=STOP_EVENT,
            reconcile_fn=_reconcile_one,
        )

        # --- Métricas institucionales de riesgo (cada 5 min) ---
        def _get_open_positions_all() -> list[dict]:
            try:
                _require_repos()
                rows = OPEN_POS_REPO.list_all()  # type: ignore[union-attr]
                return list(rows or [])
            except Exception:
                return []

        executor.submit(
            risk_metrics_worker,
            symbols=SYMBOLS,
            stop_event=STOP_EVENT,
            interval_s=300.0,
            compute_equity_total=lambda: compute_equity_total(SYMBOLS, locks),
            get_open_positions=_get_open_positions_all,
            get_live_price=get_precio_actual,
            db=DB,
        )

        # --- Health check HTTP (no bloquea trading) ---
        health_host = os.getenv('HEALTH_HOST', '0.0.0.0')
        health_port = int(os.getenv('HEALTH_PORT', '8000'))

        def _ping_binance() -> bool:
            try:
                _binance_call(client_trade.ping, tries=1)
                return True
            except Exception:
                return False

        executor.submit(
            start_health_server,
            host=health_host,
            port=health_port,
            stop_event=STOP_EVENT,
            db=DB,
            health_monitor=SYSTEM_HEALTH_MONITOR,
            global_risk=GLOBAL_RISK_CONTROLLER,
            compute_equity_total=lambda: compute_equity_total(SYMBOLS, locks),
            ping_binance=_ping_binance,
        )

        # Mantener vivo hasta stop
        while not STOP_EVENT.is_set():
            STOP_EVENT.wait(1)

    logger.info("Bot detenido")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fallo al iniciar el bot: {e}")
        raise
