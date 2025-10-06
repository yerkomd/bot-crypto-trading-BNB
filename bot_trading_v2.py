from binance.client import Client
import time
import pandas as pd
import os
import logging
import ta
from decimal import Decimal, ROUND_DOWN
import requests # Nuevo import
from dotenv import load_dotenv # Nuevo import para cargar variables de entorno
import threading
from concurrent.futures import ThreadPoolExecutor

# Cargar variables de entorno desde un archivo .env
load_dotenv()

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

# Parámetros con valores por defecto
SYMBOLS = [s.strip() for s in os.getenv('SYMBOLS', 'BTCUSDT').split(',')]  # Limpia espacios y usa default
rsi_threshold = float(os.getenv('RSI_THRESHOLD', 40))
take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', 2.0))
stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', 2.0))
trailing_take_profit_pct = float(os.getenv('TRAILING_TAKE_PROFIT_PCT', 0.8))
trailing_stop_pct = float(os.getenv('TRAILING_STOP_PCT', 0.3))
position_size = float(os.getenv('POSITION_SIZE', 0.03))
timeframe = os.getenv('TIMEFRAME', '1h')
step_size = float(os.getenv('STEP_SIZE', 0.00001))
min_notional = float(os.getenv('MIN_NOTIONAL', 10))
# Variables para control de frecuencia de compras
cooldown_seconds = int(os.getenv('BUY_COOLDOWN_SECONDS', '5400'))  # 90 min por defecto
poll_interval = int(os.getenv('POLL_INTERVAL_SECONDS', '30'))      # frecuencia de chequeo

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("Advertencia: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. Las notificaciones de Telegram no funcionarán.")
    print("Asegúrate de crear un archivo .env con las variables.")
# Configurar la conexión a Binance en Testnet
client = Client(binance_api_key, binance_api_secret, testnet=True)
# Obtener balance inicial
def get_balance():
    try:
        balance = client.get_asset_balance(asset='USDT')
        if not balance:
            return 0.0
        return float(balance.get('free', 0.0))
    except Exception as e:
        print(f"Error obteniendo balance USDT: {e}")
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
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
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

def ajustar_cantidad(cantidad, step_size):
    """
    Ajusta 'cantidad' para que sea un múltiplo de 'step_size' utilizando la precisión adecuada.
    """
    # Convertir a Decimal para mayor precisión
    cantidad_dec = Decimal(str(cantidad))
    step_size_dec = Decimal(str(step_size))
    
    # Obtener el número de decimales permitido a partir del step_size
    decimales = -step_size_dec.as_tuple().exponent
    # Crear el factor de cuantización, por ejemplo: '1e-5' para 5 decimales
    factor = Decimal('1e-' + str(decimales))
    
    # Truncar la cantidad hacia abajo para que sea múltiplo del step_size
    cantidad_ajustada = cantidad_dec.quantize(factor, rounding=ROUND_DOWN)
    return float(cantidad_ajustada)

#Calculo de metricas tecnicas para el traiding

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
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    return df

# Función para guardar logs de operaciones para análisis ML

def log_trade(timestamp, symbol, trade_type, price, amount, profit, volatility, rsi, stochrsi_k, stochrsi_d, description):
    log_filename = get_log_filename(symbol)
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write('fecha|symbol|trade_type|price|amount|profit|volatility|rsi|stochrsi_k|stochrsi_d|description\n')
    with open(log_filename, 'a') as f:
        f.write(f"{timestamp}|{symbol}|{trade_type}|{price}|{amount}|{profit}|{volatility}|{rsi}|{stochrsi_k}|{stochrsi_d}|{description}\n")

# Función para enviar notificaciones a Telegram
def send_positions_to_telegram(symbol, data, token, chat_id):
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
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

# Función para enviar notificaciones a Telegram
def send_event_to_telegram(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")
# Función para calcular el % de cambio en los últimos 5 intervalos de 4h
def market_change_last_5_intervals(symbol):
    """
    Calcula el % de cambio en cada uno de los últimos 5 intervalos de 4h.
    Devuelve una lista con el % de cambio por intervalo y el promedio total.
    """
    # Obtener las últimas 5 velas de 4h
    candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, limit=5)
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'
    ])
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)

    percent_changes = []
    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        percent_change = ((close_price - open_price) / open_price) * 100
        percent_changes.append(percent_change)

    # suma los cambios de los %, si la sumatoria es positiva, el mercado está en tendencia alcista
    sum_changes = sum(percent_changes)

    return sum_changes
# Obtener precio actual directamente desde Binance
def get_precio_actual(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error obteniendo precio para {symbol}: {e}")
        return None

# Función para ejecutar órdenes con confirmación y reintentos
def ejecutar_orden_con_confirmacion(tipo, symbol, cantidad, max_intentos=10):
    """
    Ejecuta una orden de compra o venta en Binance y espera confirmación.
    Reintenta hasta max_intentos veces. Si falla, envía alerta por Telegram.
    tipo: 'buy' o 'sell'
    """
    for intento in range(1, max_intentos + 1):
        try:
            if tipo == 'buy':
                respuesta = client.order_market_buy(symbol=symbol, quantity=cantidad)
            elif tipo == 'sell':
                respuesta = client.order_market_sell(symbol=symbol, quantity=cantidad)
            else:
                raise ValueError("Tipo de orden no válido. Usa 'buy' o 'sell'.")

            if respuesta.get('status') == 'FILLED':
                print(f"✅ Orden {tipo} exitosa de {cantidad} {symbol} (intento {intento})")
                send_event_to_telegram(
                    f"✅ Orden {tipo} exitosa de {cantidad} {symbol} (intento {intento})",
                    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                )
                return respuesta
            else:
                print(f"⚠️ Orden {tipo} no completada (intento {intento}): {respuesta}")
                time.sleep(2)
        except Exception as e:
            print(f"❌ Error en orden {tipo} (intento {intento}): {e}")
            time.sleep(2)
    # Si no se logra después de max_intentos
    alerta = f"❌ No se pudo ejecutar la orden {tipo} de {cantidad} {symbol} después de {max_intentos} intentos."
    print(alerta)
    send_event_to_telegram(alerta, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    return None

# Función para monitorear stop loss de forma asíncrona
# Esta función se ejecutará en un hilo separado para monitorear las posiciones abiertas y ejecutar stop
def monitoring_open_position(symbol, lock):
    while True:
        with lock:
            positions = load_positions(symbol)
            if not positions:
                time.sleep(60)
                continue

            data = get_data_binance(symbol)
            # Calcula métricas necesarias para el log y control
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

            for position in positions[:]:
                take_profit = position['take_profit']
                stop_loss = position['stop_loss']
                buy_price = position['buy_price']
                amount = position['amount']

                if close_price >= take_profit:
                    # Actualiza trailing take profit y stop
                    position['take_profit'] = close_price * (1 + trailing_take_profit_pct / 100)
                    position['stop_loss'] = close_price * (1 - trailing_stop_pct / 100)
                    save_positions(symbol, positions)
                else:
                    if close_price <= stop_loss:
                        # Decide TAKE PROFIT vs STOP LOSS según relación con buy_price
                        if stop_loss >= buy_price:
                            # TAKE PROFIT
                            respuesta = ejecutar_orden_con_confirmacion('sell', symbol, amount)  # descomenta si quieres trading real
                            message = f'🎯 TAKE PROFIT: Vendemos {amount:.6f} {symbol} a {close_price:.2f} USDT'
                            log_trade(timestamp, symbol, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'TAKE PROFIT')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(symbol, positions)
                        else:
                            # STOP LOSS
                            respuesta = ejecutar_orden_con_confirmacion('sell', symbol, amount)  # descomenta si quieres trading real
                            message = f'🚨 STOP LOSS: Vendemos {amount:.6f} {symbol} a {close_price:.2f} USDT'
                            log_trade(timestamp, symbol, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'STOP LOSS')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(symbol, positions)
        time.sleep(60)  # Monitorea cada minuto
# listen telegram messages
# ...

def listen_telegram_commands(token, chat_id, symbols, locks):
    last_update_id = None
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    print("Escuchando comandos de Telegram...")
    while True:
        try:
            params = {"timeout": 60}
            if last_update_id:
                params["offset"] = last_update_id + 1
            response = requests.get(url, params=params, timeout=65)
            data = response.json()
            if "result" in data:
                for update in data["result"]:
                    last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        text = update["message"]["text"].strip().lower()
                        print(f"Comando recibido: {text}")

                        if text == "/positions":
                            for symbol in symbols:
                                data = get_data_binance(symbol)
                                send_positions_to_telegram(symbol, data, token, chat_id)

                        elif text == "/sellall":
                            for symbol in symbols:
                                with locks[symbol]:
                                    positions = load_positions(symbol)
                                    data = get_data_binance(symbol)
                                    close_price = data['close'].iloc[-1]
                                    for position in positions[:]:
                                        amount = position['amount']
                                        ejecutar_orden_con_confirmacion('sell', symbol, amount)
                                        log_trade(pd.Timestamp.now(), symbol, 'sell', close_price, amount,
                                                  (close_price - position['buy_price']) * amount, 0, 0, 0, 0, 'SELL ALL')
                                        positions.remove(position)
                                    save_positions(symbol, positions)
                                    send_event_to_telegram(f"🚨 Todas las posiciones de {symbol} vendidas.", token, chat_id)

                        elif text == "/stop":
                            send_event_to_telegram("🛑 Bot detenido por comando.", token, chat_id)
                            os._exit(0)

                        elif text == "/balance" or text == "/saldo":
                            # Obtener USDT disponible (puede fallar si API no responde)
                            try:
                                usdt_free = get_balance()
                            except Exception as e:
                                usdt_free = None
                                print(f"Error obteniendo balance USDT: {e}")

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

                            msg = ""
                            if usdt_free is not None:
                                msg += f"💵 Balance disponible (USDT): {usdt_free:.2f} USDT\n"
                            else:
                                msg += "💵 Balance USDT: error al obtener\n"
                            msg += f"📦 Valor estimado posiciones abiertas: {total_positions_value:.2f} USDT\n"
                            if details:
                                msg += "\nDetalles:\n" + details

                            send_event_to_telegram(msg, token, chat_id)

            time.sleep(1)
        except Exception as e:
            print(f"Error escuchando comandos de Telegram: {e}")
            time.sleep(5)

# Ejecutar la estrategia en tiempo real
def run_strategy(symbol, lock):
    balance = get_balance()
    print(f'Balance inicial: {balance:.2f} USDT')
    n = 0
    last_buy_time = None

    while True:
        try:
            with lock:
                positions = load_positions(symbol)

            data = get_data_binance(symbol)
            df = cal_metrics_technig(data,14, 10, 20)
            close_price = df['close'].iloc[-1]
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            open_price = df['open'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            stochrsi_k = df['stochrsi_k'].iloc[-1]
            stochrsi_d = df['stochrsi_d'].iloc[-1]
            volatility = (high - low) / open_price * 100  # Calcular volatilidad en porcentaje
            timestamp = df['timestamp'].iloc[-1]
            print(f'Último precio: {close_price:.2f} {symbol} | Volatilidad: {volatility:.2f}% | rsi: {rsi:.2f} | stochrsi_k: {stochrsi_k:.2f} | stochrsi_d: {stochrsi_d:.2f}')
            # Cooldown check
            now = time.time()
            can_buy = True
            if last_buy_time is not None and (now - last_buy_time) < cooldown_seconds:
                can_buy = False        
            # Lógica de compra
            if (can_buy and rsi < rsi_threshold and 
                stochrsi_k > stochrsi_d and 
                balance > min_notional and
                len(positions) < 5): # Limitar a 5 posiciónes abiertas

                buy_amount = (balance * position_size) / close_price                
                buy_amount = ajustar_cantidad(buy_amount, step_size)                
                respuesta = ejecutar_orden_con_confirmacion('buy', symbol, buy_amount)
                if respuesta is not None:
                    new_position = {
                        'buy_price': close_price,
                        'amount': buy_amount,
                        'timestamp': timestamp,
                        'take_profit': close_price * (1 + take_profit_pct/100),
                        'stop_loss': close_price * (1 - stop_loss_pct/100)
                    }
                    with lock:
                        positions = load_positions(symbol)  # recargar por seguridad
                        positions.append(new_position)
                        save_positions(symbol, positions)
                balance = get_balance()
                last_buy_time = time.time()
                message = (f'📈 COMPRA: {buy_amount:.6f} {symbol} a {close_price:.2f} USDT')
                # Enviar notificación a Telegram
                send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                log_trade(timestamp, symbol, 'buy', close_price, buy_amount, 0, volatility, rsi, stochrsi_k, stochrsi_d, 'BUY' )

            else:
                if can_buy:
                    movimiento_mercado = market_change_last_5_intervals(symbol)
                    if movimiento_mercado <= -5 and len(positions) < 9:

                        buy_amount = (balance * position_size) / close_price                    
                        buy_amount = ajustar_cantidad(buy_amount, step_size)                    
                        respuesta = ejecutar_orden_con_confirmacion('buy', symbol, buy_amount)
                        if respuesta is not None:
                            new_position = {
                                'buy_price': close_price,
                                'amount': buy_amount,
                                'timestamp': timestamp,
                                'take_profit': close_price * (1 + take_profit_pct/100),
                                'stop_loss': close_price * (1 - stop_loss_pct/100)
                            }
                            with lock:
                                positions = load_positions(symbol)
                                positions.append(new_position)
                                save_positions(symbol, positions)
                        balance = get_balance()
                        last_buy_time = time.time()
                        message = (f'📈 COMPRA: {buy_amount:.6f} {symbol} a {close_price:.2f} USDT por caida de precio {movimiento_mercado}%')
                        # Enviar notificación a Telegram
                        send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                        log_trade(timestamp, symbol, 'buy', close_price, buy_amount, 0, volatility, rsi, stochrsi_k, stochrsi_d, 'BUY' )                

            n = n + 1
            print(n)
            #send_positions_to_telegram(symbol, data, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            if n % 6 == 0:  # Cada 6 iteraciones (3 horas aprox.)
                send_positions_to_telegram(symbol, data, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                print("Enviando posiciones abiertas a Telegram...")

            time.sleep(poll_interval)
        except Exception as e:
            print(f'Error en la ejecución: {e}')
            time.sleep(5)  # Esperar antes de reintentar

# Ejecutar estrategia en tiempo real

# En tu función principal:
if __name__ == "__main__":
    locks = {symbol: threading.Lock() for symbol in SYMBOLS}    
    with ThreadPoolExecutor(max_workers=(len(SYMBOLS) * 2) + 1) as executor:
        for symbol in SYMBOLS:
            executor.submit(monitoring_open_position, symbol, locks[symbol])
            executor.submit(run_strategy, symbol, locks[symbol])
        executor.submit(listen_telegram_commands, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, locks)
        # Mantener vivo
        executor.shutdown(wait=True)
