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

# Cargar variables de entorno desde un archivo .env
load_dotenv()

if not os.path.exists('./files'):
    os.makedirs('./files')
    
log_filename = './files/trading_log.csv'
# Archivo para guardar posiciones abiertas
positions_file = './files/open_positions.csv'

# Si el archivo no existe, escribir la cabecera con delimitador pipe
if not os.path.exists(log_filename):
    with open(log_filename, 'w') as f:
        f.write('fecha|nivel|trade_type|price|amount|profit|volatility|rsi|stochrsi_k|stochrsi_d|description\n')

# Configurar logging para registrar las operaciones en un archivo con delimitador pipe
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s|%(levelname)s|%(message)s'
)

binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_SECRET')
# --- Configuración de Telegram (NUEVO) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') # Carga desde .env
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')     # Carga desde .env

# Parámetros con valores por defecto
symbol = os.getenv('SYMBOL', 'BTCUSDT')  # Usa BTCUSDT si no está definido
rsi_threshold = float(os.getenv('RSI_THRESHOLD'))  # Default 40
take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT'))
stop_loss_pct = float(os.getenv('STOP_LOSS_PCT'))
position_size = float(os.getenv('POSITION_SIZE'))
timeframe = os.getenv('TIMEFRAME')
step_size = 0.00001000 
min_notional = 10  # Monto mínimo en USDT para abrir una posición



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

# Cargar posiciones abiertas desde CSV
def load_positions():
    global positions
    if os.path.exists(positions_file) and os.stat(positions_file).st_size > 0:
        try:
            df = pd.read_csv(positions_file)
            # Filtrar filas que tengan valores numéricos válidos
            df = df[pd.to_numeric(df['buy_price'], errors='coerce').notnull()]
            positions = clear()
            positions.extend(df.to_dict('records'))            
        except pd.errors.EmptyDataError:
            positions.clear()
    else:
        positions.clear()
    return positions

# Guardar posiciones abiertas en CSV
def save_positions(positions):
    df = pd.DataFrame(positions)
    df.to_csv(positions_file, index=False)

# Obtener datos en tiempo real
def get_data_binance():
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

    # Cálculo del Stochastic RSI
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
def log_trade(trade_type, price, amount, profit, volatility, rsi, stochrsi_k, stochrsi_d, description):
    logging.info(f"{trade_type}|{price}|{amount}|{profit}|{volatility}|{rsi}|{stochrsi_k}|{stochrsi_d}|{description}")

# Función para enviar notificaciones a Telegram
def send_positions_to_telegram(data, token, chat_id):
    global positions
    if not positions:
        message = "No hay posiciones abiertas actualmente. \n"
        # Obtener el último precio y métricas técnicas        
        df = cal_metrics_technig(data, 14, 10, 20)
        close_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        stochrsi_k = df['stochrsi_k'].iloc[-1]
        stochrsi_d = df['stochrsi_d'].iloc[-1]
        message += (
            f"\nPrecio actual: {close_price:.2f} USDT\n"
            f"RSI: {rsi:.2f}\n"
            f"StochRSI K: {stochrsi_k:.2f}\n"
            f"StochRSI D: {stochrsi_d:.2f}\n"
            f"\n*Criterio de compra:* RSI < {rsi_threshold} y StochRSI K > StochRSI D\n"
        )
        message += "\n\nEl balance actual es: {:.2f} USDT".format(get_balance())
        message += "\n\n*Esperando nuevas oportunidades de compra...*"
        message += "\n\n*¡Mantente atento a las actualizaciones!*"
    else:
        message = "📊 *Posiciones abiertas de BTC:*\n"
        for pos in positions:
            message += (
                f"- Precio compra: {pos['buy_price']:.2f} USDT\n"
                f"  Cantidad: {pos['amount']:.6f} BTC\n"
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

# Función para monitorear stop loss de forma asíncrona
# Esta función se ejecutará en un hilo separado para monitorear las posiciones abiertas y ejecutar stop
def monitor_stop_loss(client, symbol, stop_loss_pct, balance, lock):
    global positions
    while True:
        with lock:
            current_price = get_data_binance()['close'].iloc[-1]
            for position in positions[:]:
                tacke_profit = position['tacke_profit']
                stop_loss = position['stop_loss']
                buy_price = position['buy_price']
                amount = position['amount']

                if close_price >= tacke_profit:
                    position['tacke_profit'] = close_price * (1 + trailing_tacke_profit_pct/100)  # Actualizar take profit
                    position['stop_loss'] = close_price * (1 - trailing_stop_pct /100)  # Actualizar stop loss
                    save_positions(positions)
                else:
                    if close_price <= stop_loss:
                        if stop_loss >= buy_price:
                            client.order_market_sell(symbol=symbol, quantity=amount)
                            balance = get_balance()
                            message = (f'🎯 TAKE PROFIT: Vendemos {amount:.6f} BTC a {close_price:.2f} USDT')
                            log_trade(timestamp, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'TAKE PROFIT')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(positions)
                        else:
                            client.order_market_sell(symbol=symbol, quantity=amount)
                            balance = get_balance()
                            message(f'🚨 STOP LOSS: Vendemos {amount:.6f} BTC a {close_price:.2f} USDT')
                            log_trade(timestamp, 'sell', close_price, amount, (close_price - buy_price) * amount, volatility, rsi, stochrsi_k, stochrsi_d, 'STOP LOSS')
                            send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                            positions.remove(position)
                            save_positions(positions)
                    

                if current_price <= buy_price * (1 - stop_loss_pct / 100):
                    client.order_market_sell(symbol=symbol, quantity=amount)
                    balance += amount * current_price
                    message = (f'🚨 STOP LOSS ASYNC: Vendemos {amount:.6f} BTC a {current_price:.2f} USDT')
                    send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                    positions.remove(position)
                    save_positions(positions)
        time.sleep(60)  # Monitorea cada minuto

# Ejecutar la estrategia en tiempo real
def run_strategy():
    balance = get_balance()
    global positions
    print(f'Balance inicial: {balance:.2f} USDT')
    n = 0

    while True:
        try:
            data = get_data_binance()
            
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
            print(f'Último precio: {close_price:.2f} USDT | Volatilidad: {volatility:.2f}% | rsi: {rsi:.2f} | stochrsi_k: {stochrsi_k:.2f} | stochrsi_d: {stochrsi_d:.2f}')            
            # Lógica de compra
            if (rsi < rsi_threshold and 
                stochrsi_k > stochrsi_d and 
                balance > min_notional and
                len(positions) < 5): # Limitar a 5 posiciónes abiertas
                buy_amount = (balance * position_size) / close_price
                print(buy_amount)
                buy_amount = ajustar_cantidad(buy_amount, step_size)
                print(buy_amount)
                order = client.order_market_buy(symbol=symbol, quantity=buy_amount)
                new_position = {
                    'buy_price': close_price, 
                    'amount': buy_amount, 
                    'timestamp': timestamp, 
                    'tacke_profit': close_price * (1 + take_profit_pct/100),
                    'stop_loss': close_price * (1 - stop_loss_pct/100)}
                positions.append(new_position)
                balance = get_balance()
                message = (f'📈 COMPRA: {buy_amount:.6f} BTC a {close_price:.2f} USDT')
                # Enviar notificación a Telegram
                send_event_to_telegram(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                log_trade('buy', close_price, buy_amount, 0, volatility, rsi, stochrsi_k, stochrsi_d, 'BUY' )
                save_positions(positions)
 
            n = n + 1
            print(n)
            send_positions_to_telegram(data, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            # if n % 6 == 0:  # Cada 6 iteraciones (3 horas)
            #     send_positions_to_telegram(positions, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            #     print("Enviando posiciones abiertas a Telegram...")

            time.sleep(1800)  
        except Exception as e:
            print(f'Error en la ejecución: {e}')
            time.sleep(60)  # Esperar antes de reintentar

# Ejecutar estrategia en tiempo real

# En tu función principal:
if __name__ == "__main__":
    lock = threading.Lock()
    positions = load_positions()
    # Lanza el hilo de monitoreo asíncrono
    stop_loss_thread = threading.Thread(target=monitor_stop_loss, args=(client, symbol, stop_loss_pct, get_balance(), lock), daemon=True)
    stop_loss_thread.start()
    # Aquí sigue tu lógica principal (run_strategy)
    run_strategy()
