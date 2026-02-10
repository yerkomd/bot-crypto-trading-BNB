from binance.client import Client
import time
import pandas as pd
import os
import sys
import logging
import logging.handlers
import ta
from decimal import Decimal, ROUND_DOWN
import requests # Nuevo import
from dotenv import load_dotenv # Nuevo import para cargar variables de entorno
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import csv
from dataclasses import dataclass
from datetime import datetime, timezone, date

# Cargar variables de entorno desde un archivo .env (no toca red; seguro para imports/tests)
load_dotenv()

STOP_EVENT = threading.Event()

# Fallback robusto para excepciones de la lib
try:
    from binance.exceptions import BinanceAPIException
except Exception:
    try:
        from binance.error import ClientError as BinanceAPIException
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

if not os.path.exists('./files'):
    os.makedirs('./files')
    
def get_log_filename(symbol):
    return f'./files/trading_log_{symbol}.csv'

def get_positions_file(symbol):
    return f'./files/open_positions_{symbol}.csv'


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
# Variables para control de frecuencia de compras
cooldown_seconds = int(os.getenv('BUY_COOLDOWN_SECONDS', '5400'))  # 90 min por defecto
poll_interval = int(os.getenv('POLL_INTERVAL_SECONDS', '30'))      # frecuencia de chequeo

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
ATR_MULTIPLIERS = {
    "BULL": {"tp": 2.5, "sl": 1.5, "trailing_sl": 1.2},
    "LATERAL": {"tp": 2.0, "sl": 1.2, "trailing_sl": 1.0},
    "BEAR": None,
}

# Si hay balance bloqueado en órdenes abiertas, un STOP LOSS podría no poder vender.
# Si activas esta opción, el bot intentará cancelar órdenes abiertas del símbolo para liberar balance.
# 0/false = desactivado (default), 1/true = activado
cancel_open_orders_on_stop = str(os.getenv('CANCEL_OPEN_ORDERS_ON_STOP', '0')).strip().lower() in (
    '1', 'true', 'yes', 'y', 'on'
)

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.info("Advertencia: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. Las notificaciones de Telegram no funcionarán.")
    logger.info("Asegúrate de crear un archivo .env con las variables.")
# El cliente de Binance se inicializa en main() para permitir importación del módulo (tests) sin tocar red.
client = None
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
    _require_client()
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except BinanceAPIException as be:
            code = getattr(be, 'code', None)
            msg = getattr(be, 'message', str(be))
            # no reintentar errores típicamente definitivos
            # -2015: Invalid API-key, IP, or permissions for action
            if str(code) in ('-2015', '-2010', '2010') or 'LOT_SIZE' in str(msg) or 'MIN_NOTIONAL' in str(msg):
                raise
            if attempt >= tries:
                raise
            sleep_s = (backoff_base ** (attempt - 1))
            logger.warning(f"Binance: error intento={attempt}/{tries} code={code} msg={msg}. Reintentando en {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as e:
            if attempt >= tries:
                raise
            sleep_s = (backoff_base ** (attempt - 1))
            logger.warning(f"Binance: error intento={attempt}/{tries} {e}. Reintentando en {sleep_s:.1f}s")
            time.sleep(sleep_s)

def _require_client():
    if client is None:
        raise RuntimeError("Binance client no inicializado. Ejecuta el script vía __main__.")

def _validate_required_env():
    missing = []
    if not os.getenv('BINANCE_API_KEY'):
        missing.append('BINANCE_API_KEY')
    if not os.getenv('BINANCE_API_SECRET'):
        missing.append('BINANCE_API_SECRET')
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
        balance = _binance_call(client.get_asset_balance, asset='USDT')
        if not balance:
            return 0.0
        return float(balance.get('free', 0.0))
    except Exception as e:
        logger.error(f"Error obteniendo balance USDT: {e}")
        return 0.0
# Cargar posiciones abiertas desde CSV por símbolo
def load_positions(symbol):
    positions_file = get_positions_file(symbol)
    if os.path.exists(positions_file) and os.stat(positions_file).st_size > 0:
        try:
            df = pd.read_csv(positions_file)
            df = df[pd.to_numeric(df['buy_price'], errors='coerce').notnull()]
            return df.to_dict('records')
        except pd.errors.EmptyDataError:
            return []
    return []


# Guardar posiciones abiertas en CSV
def save_positions(symbol, positions_list):
    positions_file = get_positions_file(symbol)
    tmp = positions_file + '.tmp'
    df = pd.DataFrame(positions_list)
    # escribir temporalmente y renombrar (atomic)
    df.to_csv(tmp, index=False)
    os.replace(tmp, positions_file)

# Obtener datos en tiempo real
def get_data_binance(symbol, interval='1h', limit=40):
    """
    Devuelve DataFrame con las últimas `limit` velas para `symbol`.
    Usa interval como string ('1h', '4h', '15m', etc.) para evitar problemas con constantes.
    """
    candles = _binance_call(client.get_klines, symbol=symbol, interval=interval, limit=limit)
    
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convertir timestamps a fechas
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)    
    return df

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
        df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    except Exception:
        df['ema200'] = pd.NA

    # EMA50 (tendencia intermedia para filtro de entradas)
    try:
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    except Exception:
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

# Función para guardar logs de operaciones para análisis ML

def log_trade(timestamp, symbol, trade_type, price, amount, profit, volatility, rsi, stochrsi_k, stochrsi_d, description, extra: dict | None = None):
    log_filename = get_log_filename(symbol)
    fieldnames = [
        'fecha', 'symbol', 'trade_type', 'price', 'amount', 'profit', 'volatility',
        'rsi', 'stochrsi_k', 'stochrsi_d', 'description',
        # Contexto de estrategia (puede ser vacío en registros antiguos)
        'regime',
        'rsi_threshold_used',
        'take_profit_pct_used',
        'stop_loss_pct_used',
        'trailing_tp_pct_used',
        'trailing_sl_pct_used',
        'buy_cooldown_used',
        'position_size_used',
    ]

    delimiter = ','
    if os.path.exists(log_filename) and os.stat(log_filename).st_size > 0:
        try:
            with open(log_filename, 'r', newline='') as rf:
                header = (rf.readline() or '').strip()
            if '|' in header and ',' not in header:
                delimiter = '|'
        except Exception:
            delimiter = ','

    file_exists = os.path.exists(log_filename) and os.stat(log_filename).st_size > 0
    extra = extra or {}
    row = {
        'fecha': str(timestamp),
        'symbol': symbol,
        'trade_type': trade_type,
        'price': price,
        'amount': amount,
        'profit': profit,
        'volatility': volatility,
        'rsi': rsi,
        'stochrsi_k': stochrsi_k,
        'stochrsi_d': stochrsi_d,
        'description': description,

        'regime': extra.get('regime'),
        'rsi_threshold_used': extra.get('rsi_threshold_used'),
        'take_profit_pct_used': extra.get('take_profit_pct_used'),
        'stop_loss_pct_used': extra.get('stop_loss_pct_used'),
        'trailing_tp_pct_used': extra.get('trailing_tp_pct_used'),
        'trailing_sl_pct_used': extra.get('trailing_sl_pct_used'),
        'buy_cooldown_used': extra.get('buy_cooldown_used'),
        'position_size_used': extra.get('position_size_used'),
    }

    with open(log_filename, 'a', newline='') as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, delimiter=delimiter)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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
    # Obtener las últimas 5 velas de 4h
    candles = _binance_call(client.get_klines, symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, limit=5)
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'
    ])
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)

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
        ticker = _binance_call(client.get_symbol_ticker, symbol=symbol)
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
    side_fn = client.order_market_buy if tipo == 'buy' else client.order_market_sell

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
                    status_info = _binance_call(client.get_order, symbol=symbol, orderId=order_id, tries=2)
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
                    break  # salir de verificación e ir a reintento

                # NEW u otros: seguir
                logger.debug(f"[{symbol}] Estado {status} verif {v}/{verificaciones_por_intento}")

            # Si terminó verificación sin FILLED
            if last_status in ('NEW', 'PARTIALLY_FILLED'):
                # Intentar cancelar para evitar que quede colgada
                try:
                    _binance_call(client.cancel_order, symbol=symbol, orderId=order_id, tries=2)
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
    _require_client()

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
            data = get_data_binance(symbol, interval=timeframe)
            df_metrics = cal_metrics_technig(data.copy(), 14, 10, 20)
            close_price = df_metrics['close'].iloc[-1]
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
                equity_est = float(get_balance()) + _positions_current_value(positions, float(close_price))
            except Exception:
                equity_est = float(get_balance())

            # Alimentar risk manager con métricas intradía
            try:
                RISK_MANAGER.observe(symbol, positions=positions, price=float(close_price), equity_total=float(equity_est))
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
                if (not trailing_active) and close_price >= tp_initial:
                    trailing_active = True
                    to_update.append({
                        'match': {'buy_price': buy_price, 'timestamp': str(position.get('timestamp'))},
                        'fields': {
                            'trailing_active': True,
                            'tp_initial': tp_initial,
                            'max_price': max(float(position.get('max_price', buy_price) or buy_price), float(close_price)),
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
                        changed = update_trailing_stop_atr(tmp, float(close_price), float(atr_now), float(trailing_mult))
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
                        candidate = float(close_price) * (1 - trailing_sl_pct_pos / 100)
                        if candidate > stop_loss:
                            updated_fields['stop_loss'] = candidate
                            updated_fields['max_price'] = max(float(position.get('max_price', buy_price) or buy_price), float(close_price))
                            updated_fields['trailing_active'] = True
                            updated_fields['tp_initial'] = tp_initial

                    if updated_fields:
                        if 'stop_loss' in updated_fields:
                            try:
                                effective_stop_loss = float(updated_fields['stop_loss'])
                            except Exception:
                                effective_stop_loss = stop_loss
                        to_update.append({
                            'match': {'buy_price': buy_price, 'timestamp': str(position.get('timestamp'))},
                            'fields': updated_fields
                        })

                # Venta por stop
                # Nota: stop_loss puede ser actualizado bajo lock; aquí usamos el valor actual.
                if close_price <= effective_stop_loss:
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
                            bp = float(upd['match']['buy_price'])
                            ts = str(upd['match']['timestamp'])
                            for p in current:
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
                sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, close_price, for_sell=True, filters=symbol_filters)

                # Si el balance está bloqueado en órdenes abiertas, un STOP LOSS puede fallar.
                if sell_qty is None and cancel_open_orders_on_stop and locked_qty > 0:
                    canceled = cancel_all_open_orders(symbol)
                    if canceled > 0:
                        bal = get_base_asset_balance(symbol)
                        free_qty = float(bal.get('free', 0.0) or 0.0)
                        locked_qty = float(bal.get('locked', 0.0) or 0.0)
                        sell_qty_raw = min(requested_qty, free_qty)
                        sell_qty, motivo = sanitize_quantity(symbol, sell_qty_raw, close_price, for_sell=True, filters=symbol_filters)

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
                msg = f'{emoji} {reason}: Vendemos {sell_qty:.6f} {symbol} a {close_price:.4f} USDT'
                send_event_to_telegram(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                # registrar pnl realizado en risk manager
                realized = (close_price - float(removed.get('buy_price', 0.0) or 0.0)) * float(sell_qty)
                try:
                    RISK_MANAGER.record_realized_pnl(symbol, realized, timestamp, equity_total=equity_est)
                except Exception:
                    pass
                log_trade(
                    timestamp, symbol, 'sell', close_price, sell_qty,
                    (close_price - removed['buy_price']) * sell_qty,
                    volatility, rsi, stochrsi_k, stochrsi_d, reason,
                    extra={
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
                            data_symbol = get_data_binance(symbol)
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
    info = _binance_call(client.get_symbol_info, symbol)
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
    info = _binance_call(client.get_symbol_info, symbol)
    base = info['baseAsset']
    try:
        bal = _binance_call(client.get_asset_balance, asset=base)
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
    info = _binance_call(client.get_symbol_info, symbol)
    base = info['baseAsset']
    try:
        bal = _binance_call(client.get_asset_balance, asset=base)
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
        orders = _binance_call(client.get_open_orders, symbol=symbol, tries=2)
        if not orders:
            return 0
        canceled = 0
        for o in orders:
            oid = o.get('orderId')
            if oid is None:
                continue
            try:
                _binance_call(client.cancel_order, symbol=symbol, orderId=oid, tries=2)
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

        if price > ema200 and adx >= 25:
            return "BULL"
        elif price < ema200 and adx >= 25:
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

# Ejecutar la estrategia en tiempo real
def run_strategy(symbol, lock):
    logger.info(f"[{symbol}] Iniciando hilo de estrategia")
    _require_client()
    balance = get_balance()
    logger.info(f"[{symbol}] Balance inicial: {balance:.2f} USDT")
    n = 0
    last_buy_time = None
    positions_cache = []

    # Cachear filtros por símbolo para evitar llamadas repetidas
    try:
        symbol_filters = get_symbol_filters(symbol)
    except Exception as e:
        logger.warning(f"[{symbol}] No se pudieron obtener filtros de símbolo: {e}")
        symbol_filters = None

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

            # Pedimos más histórico para poder calcular EMA200/ADX y detectar régimen.
            data = get_data_binance(symbol, interval=timeframe, limit=260)
            df = cal_metrics_technig(data, 14, 10, 20)
            # Usar SOLO la última vela cerrada para señales (evita lookahead)
            close_price = df['close'].iloc[-2]
            high = df['high'].iloc[-2]
            low = df['low'].iloc[-2]
            open_price = df['open'].iloc[-2]
            rsi = df['rsi'].iloc[-2]
            stochrsi_k = df['stochrsi_k'].iloc[-2]
            stochrsi_d = df['stochrsi_d'].iloc[-2]
            ema200 = df['ema200'].iloc[-2] if 'ema200' in df.columns else pd.NA
            ema50 = df['ema50'].iloc[-2] if 'ema50' in df.columns else pd.NA
            atr = df['atr'].iloc[-2] if 'atr' in df.columns else pd.NA
            volatility = (high - low) / open_price * 100 if open_price else 0
            timestamp = df['timestamp'].iloc[-2]

            # Guardas: si indicadores aún no están disponibles, no operar
            if pd.isna(rsi) or pd.isna(stochrsi_k) or pd.isna(stochrsi_d) or pd.isna(ema200) or pd.isna(ema50) or pd.isna(atr):
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

            logger.info(f"[{symbol}] Precio: {close_price:.4f} | Vol: {volatility:.2f}% | RSI: {rsi:.2f} | Stoch K/D: {stochrsi_k:.2f}/{stochrsi_d:.2f} | Posiciones: {len(positions)}")

            # Detectar régimen usando última vela cerrada (evita lookahead)
            regime = detect_market_regime(df.iloc[:-1])
            params = REGIME_PARAMS.get(regime, LATERAL)
            cooldown_seconds_active = int(params["BUY_COOLDOWN"])
            position_size_active = float(params["POSITION_SIZE"])

            # Filtro crítico: en BEAR NO abrir nuevas posiciones (ni DCA)
            if str(regime).upper() == "BEAR":
                # Aun así alimentamos risk manager para métricas (sin tocar ejecución)
                try:
                    equity_est = float(get_balance()) + _positions_current_value(positions, float(close_price))
                    RISK_MANAGER.observe(symbol, positions=positions, price=float(close_price), equity_total=float(equity_est))
                except Exception:
                    pass
                logger.info(f"[{symbol}] Régimen BEAR: compras deshabilitadas. Solo gestión de posiciones abiertas.")
                STOP_EVENT.wait(poll_interval)
                continue

            # Cooldown
            now = time.time()
            can_buy = not (last_buy_time and (now - last_buy_time) < cooldown_seconds_active)

            # Refrescar balance solo cuando sea relevante (evita llamadas constantes)
            if can_buy:
                balance = get_balance()

            # Preferir min_notional real del símbolo si está disponible
            symbol_min_notional = (symbol_filters or {}).get('min_notional', min_notional)

            # Alimentar risk manager (métricas intradía) con equity estimado
            try:
                equity_est = float(balance) + _positions_current_value(positions, float(close_price))
                RISK_MANAGER.observe(symbol, positions=positions, price=float(close_price), equity_total=float(equity_est))
            except Exception:
                pass

            # Entradas más restrictivas (calidad > cantidad):
            # price > EMA200, EMA50 > EMA200, RSI en rango [35,55]
            entry_ok = (float(close_price) > float(ema200)) and (float(ema50) > float(ema200)) and (RSI_CONFIRM_MIN <= float(rsi) <= RSI_CONFIRM_MAX)

            # Intento compra principal
            executed = False

            if (can_buy and entry_ok and balance > symbol_min_notional and len(positions) < 5):

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
                qty, motivo = preparar_cantidad(symbol, capital_usar, close_price, filters=symbol_filters)
                if qty is None:
                    logger.info(f"[{symbol}] Skip compra (condición principal): {motivo}")
                else:
                    debug_symbol_filters(symbol)

                    # TP/SL dinámicos por ATR según régimen
                    mults = ATR_MULTIPLIERS.get(regime) or ATR_MULTIPLIERS['LATERAL']
                    tp_mult = float(mults['tp'])
                    sl_mult = float(mults['sl'])
                    trailing_sl_mult = float(mults['trailing_sl'])
                    tp_price, sl_price = calculate_atr_levels(float(close_price), float(atr), tp_mult, sl_mult)

                    resp = ejecutar_orden_con_confirmacion('buy', symbol, qty)
                    if resp:
                        new_position = {
                            'buy_price': close_price,
                            'amount': qty,
                            'timestamp': timestamp,
                            # TP/SL por ATR (compatibles con campos existentes)
                            'take_profit': tp_price,
                            'stop_loss': sl_price,

                            # Congelar parámetros por posición
                            'regime': regime,
                            'rsi_threshold': f"{RSI_CONFIRM_MIN}-{RSI_CONFIRM_MAX}",

                            # Persistir ATR y multiplicadores
                            'atr_entry': float(atr),
                            'tp_atr_mult': tp_mult,
                            'sl_atr_mult': sl_mult,
                            'trailing_sl_atr_mult': trailing_sl_mult,
                            'tp_initial': tp_price,
                            'trailing_active': False,
                            'max_price': float(close_price),
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
                            volatility, rsi, stochrsi_k, stochrsi_d, 'BUY',
                            extra={
                                'regime': regime,
                                'rsi_threshold_used': f"{RSI_CONFIRM_MIN}-{RSI_CONFIRM_MAX}",
                                # Reutilizamos columnas existentes para guardar multipliers ATR
                                'take_profit_pct_used': tp_mult,
                                'stop_loss_pct_used': sl_mult,
                                'trailing_tp_pct_used': None,
                                'trailing_sl_pct_used': trailing_sl_mult,
                                'buy_cooldown_used': cooldown_seconds_active,
                                'position_size_used': position_size_active,
                            }
                        )
                        executed = True

            # DCA (opcional y desactivado por defecto). Nunca en BEAR (ya filtrado arriba).
            # Nota: se deja como feature togglable para no introducir martingala implícita.
            if ENABLE_DCA and (not executed) and can_buy and regime in ("BULL", "LATERAL"):
                # Mantener el comportamiento antiguo solo si se habilita explícitamente.
                movimiento = market_change_last_5_intervals(symbol)
                if movimiento is not None and balance > symbol_min_notional and movimiento <= -5 and len(positions) < 9:
                    logger.warning(f"[{symbol}] DCA habilitado: movimiento={movimiento:.2f}% (controla el riesgo).")
            
            STOP_EVENT.wait(poll_interval)

        except Exception as e:
            logger.exception(f"[{symbol}] Error en run_strategy")
            STOP_EVENT.wait(5)

def main():
    global client

    # Validaciones y configuración
    _validate_required_env()

    testnet_env = (os.getenv('BINANCE_TESTNET', 'true') or 'true').strip().lower()
    use_testnet = testnet_env in ('1', 'true', 'yes', 'y', 'on')

    # Señales para detener ordenadamente
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    # Inicializar cliente Binance
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=use_testnet)
    logger.info("Cliente Binance inicializado. testnet=%s", use_testnet)

    # Validación temprana de credenciales/permisos/IP (falla rápido si hay -2015)
    try:
        _binance_call(client.get_account, tries=1)
        logger.info("Credenciales OK: acceso a cuenta verificado.")
    except BinanceAPIException as be:
        code = getattr(be, 'code', None)
        msg = getattr(be, 'message', str(be))
        if str(code) == '-2015':
            logger.error(
                "Binance auth falló (-2015): %s. Revisa: (1) BINANCE_TESTNET coincide con tu API key, "
                "(2) permisos habilitados (Read/Spot Trading), (3) restricción de IP/whitelist, "
                "(4) API key/secret sin espacios ni comillas.",
                msg,
            )
        raise

    locks = {symbol: threading.Lock() for symbol in SYMBOLS}
    logger.info(f"Iniciando bot con símbolos: {SYMBOLS}")

    with ThreadPoolExecutor(max_workers=(len(SYMBOLS) * 2) + 1) as executor:
        for symbol in SYMBOLS:
            executor.submit(monitoring_open_position, symbol, locks[symbol])
            executor.submit(run_strategy, symbol, locks[symbol])
        executor.submit(listen_telegram_commands, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, locks)

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
