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
SYMBOLS = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')  # Usa BTCUSDT si no está definido
rsi_threshold = float(os.getenv('RSI_THRESHOLD'))  # Default 40
take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT'))
stop_loss_pct = float(os.getenv('STOP_LOSS_PCT'))
trailing_take_profit_pct = float(os.getenv('TRAILING_TAKE_PROFIT_PCT'))
trailing_stop_pct = float(os.getenv('TRAILING_STOP_PCT'))
position_size = float(os.getenv('POSITION_SIZE'))
timeframe = os.getenv('TIMEFRAME')
step_size = float(os.getenv('STEP_SIZE', 0.00001))
min_notional = float(os.getenv('MIN_NOTIONAL', 10))



# Variable Globales
positions = []

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("Advertencia: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. Las notificaciones de Telegram no funcionarán.")
    print("Asegúrate de crear un archivo .env con las variables.")

# Configurar la conexión a Binance en Testnet
client = Client(binance_api_key, binance_api_secret, testnet=True)

# Obtener balance inicial
def get_balance():
    balance = client.get_asset_balance(asset='USDT')
    return float(balance['free'])

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
    df = pd.DataFrame(positions_list)
    df.to_csv(positions_file, index=False)

# Obtener datos en tiempo real
def get_data_binance(symbol):
    candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=40)
    
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
        message = "📊 *Posiciones abiertas de {symbol}:*\n"
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

# Función para monitorear stop loss de forma asíncrona
# Esta función se ejecutará en un hilo separado para monitorear las posiciones abiertas y ejecutar stop
def monitor_stop_loss(symbol, lock):
    while True:
        with lock:
            positions = load_positions(symbol)
            data = get_data_binance(symbol)
            close_price = data['close'].iloc[-1]
            for position in positions[:]:
                tacke_profit = position['tacke_profit']
                stop_loss = position['stop_loss']
                buy_price = position['buy_price']
                amount = position['amount']

                if close_price >= tacke_profit:
                    position['tacke_profit'] = close_price * (1 + trailing_take_profit_pct/100)  # Actualizar take profit
                    position['stop_loss'] = close_price * (1 - trailing_stop_pct /100)  # Actualizar stop loss
                    save_positions(symbol, positions)
                else:
                    if close_price <= stop_loss:
                        if stop_loss >= buy_price:
                            #client.order_market_sell(symbol=symbol, quantity=amount)
                            balance = get_balance()
                            message = (f'🎯 TAKE PROFIT: Vendemos {amount:.6f} {symbol} a {close_price:.2f} USDT')
                            log_trade(timestamp, symbol, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'TAKE PROFIT')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(symbol, positions)
                        else:
                            #client.order_market_sell(symbol=symbol, quantity=amount)
                            balance = get_balance()
                            message(f'🚨 STOP LOSS: Vendemos {amount:.6f} {symbol} a {close_price:.2f} USDT')
                            log_trade(timestamp, symbol, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'STOP LOSS')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(symbol, positions)
        time.sleep(60)  # Monitorea cada minuto

# Ejecutar la estrategia en tiempo real
def run_strategy(symbol, lock):
    balance = get_balance()
    print(f'Balance inicial: {balance:.2f} USDT')
    n = 0
    while True:
        try:
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
            # Lógica de compra
            if (rsi < rsi_threshold and 
                stochrsi_k > stochrsi_d and 
                balance > min_notional and
                len(positions) < 5): # Limitar a 5 posiciónes abiertas
                buy_amount = (balance * position_size) / close_price                
                buy_amount = ajustar_cantidad(buy_amount, step_size)                
                client.order_market_buy(symbol=symbol, quantity=buy_amount)
                new_position = {
                    'buy_price': close_price,
                    'amount': buy_amount,
                    'timestamp': timestamp,
                    'tacke_profit': close_price * (1 + take_profit_pct/100),
                    'stop_loss': close_price * (1 - stop_loss_pct/100)}
                positions.append(new_position)
                balance = get_balance()
                message = (f'📈 COMPRA: {buy_amount:.6f} {symbol} a {close_price:.2f} USDT')
                # Enviar notificación a Telegram
                send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                log_trade(timestamp, symbol, 'buy', close_price, buy_amount, 0, volatility, rsi, stochrsi_k, stochrsi_d, 'BUY' )
                save_positions(symbol, positions)
                time.sleep(5400)
            else:
                movimiento_mercado = market_change_last_5_intervals(symbol)
                if movimiento_mercado <= -5 and len(positions) < 9:

                    buy_amount = (balance * position_size) / close_price                    
                    buy_amount = ajustar_cantidad(buy_amount, step_size)                    
                    client.order_market_buy(symbol=symbol, quantity=buy_amount)
                    new_position = {
                        'buy_price': close_price, 
                        'amount': buy_amount, 
                        'timestamp': timestamp, 
                        'tacke_profit': close_price * (1 + take_profit_pct/100),
                        'stop_loss': close_price * (1 - stop_loss_pct/100)}
                    positions.append(new_position)
                    balance = get_balance()
                    message = (f'📈 COMPRA: {buy_amount:.6f} {symbol} a {close_price:.2f} USDT por caida de precio {movimiento_mercado}%')
                    # Enviar notificación a Telegram
                    send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                    log_trade(timestamp, symbol, 'buy', close_price, buy_amount, 0, volatility, rsi, stochrsi_k, stochrsi_d, 'BUY' )                
                    save_positions(symbol, positions)
                    time.sleep(5400)

            n = n + 1
            print(n)
            send_positions_to_telegram(symbol, data, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            # if n % 6 == 0:  # Cada 6 iteraciones (3 horas)
            #     send_positions_to_telegram(positions, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            #     print("Enviando posiciones abiertas a Telegram...")

            time.sleep(1000)
        except Exception as e:
            print(f'Error en la ejecución: {e}')
            time.sleep(10)  # Esperar antes de reintentar

# Ejecutar estrategia en tiempo real

# En tu función principal:
if __name__ == "__main__":
    locks = {symbol: threading.Lock() for symbol in SYMBOLS}
    with ThreadPoolExecutor(max_workers=len(SYMBOLS) * 2) as executor:
        for symbol in SYMBOLS:
            executor.submit(monitor_stop_loss, symbol, locks[symbol])
            executor.submit(run_strategy, symbol, locks[symbol])
        # Mantener vivo
        executor.shutdown(wait=True)
