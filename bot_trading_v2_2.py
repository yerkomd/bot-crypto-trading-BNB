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
            if str(code) in ('-2010', '2010') or 'LOT_SIZE' in str(msg) or 'MIN_NOTIONAL' in str(msg):
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

    # ADX (fuerza de tendencia)
    try:
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_indicator.adx()
    except Exception:
        df['adx'] = pd.NA

    return df

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
            f"\n*Criterio de compra:* RSI < {rsi_threshold} y StochRSI K > StochRSI D\n"
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

            # 3) Preparar actualizaciones y ventas fuera del lock
            to_update = []
            to_sell = []

            for position in positions:
                take_profit = position['take_profit']
                stop_loss = position['stop_loss']
                buy_price = position['buy_price']
                amount = position['amount']

                # Trailing por posición (solo si fue creada con parámetros por régimen)
                # Si no existe, mantiene el comportamiento anterior usando variables globales.
                try:
                    trailing_tp = float(position.get('trailing_tp_pct', trailing_take_profit_pct))
                except Exception:
                    trailing_tp = trailing_take_profit_pct
                try:
                    trailing_sl = float(position.get('trailing_sl_pct', trailing_stop_pct))
                except Exception:
                    trailing_sl = trailing_stop_pct

                if close_price >= take_profit:
                    # actualizar trailing objetivos (se aplicará bajo lock)
                    to_update.append({
                        'match': {'buy_price': buy_price, 'timestamp': str(position['timestamp'])},
                        'fields': {
                            'take_profit': close_price * (1 + trailing_tp / 100),
                            'stop_loss': close_price * (1 - trailing_sl / 100),
                        }
                    })
                elif close_price <= stop_loss:
                    to_sell.append({
                        'position': position,
                        'reason': 'TAKE PROFIT' if stop_loss >= buy_price else 'STOP LOSS'
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
                free_qty = get_free_base_asset(symbol)
                sell_qty = min(removed['amount'], free_qty)
                sell_qty, motivo = sanitize_quantity(symbol, sell_qty, close_price, for_sell=True)
                if not sell_qty:
                    logger.warning(f"[{symbol}] No se puede vender ({reason}) qty={removed['amount']} motivo={motivo}")
                    send_event_to_telegram(f"⚠️ No se puede vender {symbol} ({reason}) qty={removed['amount']} motivo={motivo}", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
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
                            for position in positions:
                                free_qty = get_free_base_asset(symbol)
                                sell_qty = min(position['amount'], free_qty)
                                sell_qty, motivo = sanitize_quantity(symbol, sell_qty, price, for_sell=True)
                                if not sell_qty:
                                    logger.info(f"[{symbol}] Skip sellall qty={position['amount']} motivo={motivo}")
                                    continue
                                resp = ejecutar_orden_con_confirmacion('sell', symbol, sell_qty)
                                if resp:
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
    "RSI_THRESHOLD": 45,
    "TAKE_PROFIT_PCT": 5,
    "STOP_LOSS_PCT": 2,
    "TRAILING_TP_PCT": 1.2,
    "TRAILING_SL_PCT": 1.0,
    "BUY_COOLDOWN": 7200,      # 2h
    "POSITION_SIZE": 0.04
}

BEAR = {
    "RSI_THRESHOLD": 30,
    "TAKE_PROFIT_PCT": 2,
    "STOP_LOSS_PCT": 1.2,
    "TRAILING_TP_PCT": 0.8,
    "TRAILING_SL_PCT": 0.6,
    "BUY_COOLDOWN": 21600,     # 6h
    "POSITION_SIZE": 0.015
}

LATERAL = {
    "RSI_THRESHOLD": 40,
    "TAKE_PROFIT_PCT": 3,
    "STOP_LOSS_PCT": 1.5,
    "TRAILING_TP_PCT": 0.9,
    "TRAILING_SL_PCT": 0.8,
    "BUY_COOLDOWN": 14400,     # 4h
    "POSITION_SIZE": 0.03
}

REGIME_PARAMS = {
    "BULL": BULL,
    "BEAR": BEAR,
    "LATERAL": LATERAL,
}

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
            close_price = df['close'].iloc[-1]
            high = df['high'].iloc[-2]
            low = df['low'].iloc[-2]
            open_price = df['open'].iloc[-2]
            rsi = df['rsi'].iloc[-2]
            stochrsi_k = df['stochrsi_k'].iloc[-2]
            stochrsi_d = df['stochrsi_d'].iloc[-2]
            volatility = (high - low) / open_price * 100 if open_price else 0
            timestamp = df['timestamp'].iloc[-2]

            # Guardas: si indicadores aún no están disponibles, no operar
            if pd.isna(rsi) or pd.isna(stochrsi_k) or pd.isna(stochrsi_d):
                logger.warning(f"[{symbol}] Indicadores no disponibles (NaN). Omitiendo ciclo.")
                STOP_EVENT.wait(poll_interval)
                continue

            logger.info(f"[{symbol}] Precio: {close_price:.4f} | Vol: {volatility:.2f}% | RSI: {rsi:.2f} | Stoch K/D: {stochrsi_k:.2f}/{stochrsi_d:.2f} | Posiciones: {len(positions)}")

            # Detectar régimen usando última vela cerrada (evita lookahead)
            regime = detect_market_regime(df.iloc[:-1])
            params = REGIME_PARAMS.get(regime, LATERAL)
            rsi_threshold_active = float(params["RSI_THRESHOLD"])
            take_profit_pct_active = float(params["TAKE_PROFIT_PCT"])
            stop_loss_pct_active = float(params["STOP_LOSS_PCT"])
            trailing_tp_pct_active = float(params["TRAILING_TP_PCT"])
            trailing_sl_pct_active = float(params["TRAILING_SL_PCT"])
            cooldown_seconds_active = int(params["BUY_COOLDOWN"])
            position_size_active = float(params["POSITION_SIZE"])

            # Cooldown
            now = time.time()
            can_buy = not (last_buy_time and (now - last_buy_time) < cooldown_seconds_active)

            # Refrescar balance solo cuando sea relevante (evita llamadas constantes)
            if can_buy:
                balance = get_balance()

            # Preferir min_notional real del símbolo si está disponible
            symbol_min_notional = (symbol_filters or {}).get('min_notional', min_notional)

            # Intento compra principal
            executed = False
            if (can_buy and rsi < rsi_threshold_active and stochrsi_k > stochrsi_d
                and balance > symbol_min_notional and len(positions) < 5):

                capital_usar = balance * position_size_active
                qty, motivo = preparar_cantidad(symbol, capital_usar, close_price, filters=symbol_filters)
                if qty is None:
                    logger.info(f"[{symbol}] Skip compra (condición principal): {motivo}")
                else:
                    debug_symbol_filters(symbol)
                    resp = ejecutar_orden_con_confirmacion('buy', symbol, qty)
                    if resp:
                        new_position = {
                            'buy_price': close_price,
                            'amount': qty,
                            'timestamp': timestamp,
                            'take_profit': close_price * (1 + take_profit_pct_active / 100),
                            'stop_loss': close_price * (1 - stop_loss_pct_active / 100),
                            # Congelar parámetros por posición (solo para posiciones nuevas)
                            'regime': regime,
                            'rsi_threshold': rsi_threshold_active,
                            'trailing_tp_pct': trailing_tp_pct_active,
                            'trailing_sl_pct': trailing_sl_pct_active,
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
                                'rsi_threshold_used': rsi_threshold_active,
                                'take_profit_pct_used': take_profit_pct_active,
                                'stop_loss_pct_used': stop_loss_pct_active,
                                'trailing_tp_pct_used': trailing_tp_pct_active,
                                'trailing_sl_pct_used': trailing_sl_pct_active,
                                'buy_cooldown_used': cooldown_seconds_active,
                                'position_size_used': position_size_active,
                            }
                        )
                        executed = True

            # Compra por caída (DCA/reactiva)
            if (not executed) and can_buy:
                movimiento = market_change_last_5_intervals(symbol)
                if movimiento is not None and balance > symbol_min_notional and movimiento <= -5 and len(positions) < 9:
                    capital_usar = balance * position_size_active
                    qty, motivo = preparar_cantidad(symbol, capital_usar, close_price, filters=symbol_filters)
                    if qty is None:
                        logger.info(f"[{symbol}] Skip compra caída: {motivo}")
                    else:
                        debug_symbol_filters(symbol)
                        resp = ejecutar_orden_con_confirmacion('buy', symbol, qty)
                        if resp:
                            new_position = {
                                'buy_price': close_price,
                                'amount': qty,
                                'timestamp': timestamp,
                                'take_profit': close_price * (1 + take_profit_pct_active / 100),
                                'stop_loss': close_price * (1 - stop_loss_pct_active / 100),
                                # Congelar parámetros por posición (solo para posiciones nuevas)
                                'regime': regime,
                                'rsi_threshold': rsi_threshold_active,
                                'trailing_tp_pct': trailing_tp_pct_active,
                                'trailing_sl_pct': trailing_sl_pct_active,
                            }
                            with lock:
                                current = load_positions(symbol)
                                current.append(new_position)
                                save_positions(symbol, current)
                            balance = get_balance()
                            last_buy_time = time.time()
                            send_event_to_telegram(f'📉 DCA {symbol}: {qty:.6f} @ {close_price:.4f} caída {movimiento:.2f}%', TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            log_trade(
                                timestamp, symbol, 'buy', close_price, qty, 0,
                                volatility, rsi, stochrsi_k, stochrsi_d, 'BUY_DCA',
                                extra={
                                    'regime': regime,
                                    'rsi_threshold_used': rsi_threshold_active,
                                    'take_profit_pct_used': take_profit_pct_active,
                                    'stop_loss_pct_used': stop_loss_pct_active,
                                    'trailing_tp_pct_used': trailing_tp_pct_active,
                                    'trailing_sl_pct_used': trailing_sl_pct_active,
                                    'buy_cooldown_used': cooldown_seconds_active,
                                    'position_size_used': position_size_active,
                                }
                            )
            
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
