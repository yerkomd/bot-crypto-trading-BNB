import os
import time
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from binance.client import Client
import requests

load_dotenv()

# --- Config / Cliente ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

# Risk multipliers por perfil (fracción del balance en USDT a usar por posición)
RISK_MULTIPLIERS = {
    'high': 0.05,    # arriesgar ~5% del balance por operación (alto riesgo)
    'medium': 0.02,  # ~2%
    'low': 0.01      # ~1%
}

# Mapeo de símbolos a perfil (se puede leer de .env como "SYMBOL_RISKS=BTCUSDT:high,ETHUSDT:medium")
def load_symbol_risks(env_key='SYMBOL_RISKS'):
    raw = os.getenv(env_key, '')
    mapping = {}
    if raw:
        for part in raw.split(','):
            if ':' in part:
                s, p = part.split(':', 1)
                mapping[s.strip().upper()] = p.strip().lower()
    return mapping

SYMBOL_RISKS = load_symbol_risks()

# util: ajustar cantidad al step_size
def ajustar_cantidad(cantidad, step_size):
    cantidad_dec = Decimal(str(cantidad))
    step_size_dec = Decimal(str(step_size))
    decimales = -step_size_dec.as_tuple().exponent
    factor = Decimal('1e-' + str(decimales))
    cantidad_ajustada = cantidad_dec.quantize(factor, rounding=ROUND_DOWN)
    return float(cantidad_ajustada)

# Obtener precio directo
def get_precio_actual(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception:
        return None

# Tamaño de posición recomendado en cantidad de asset (no en USDT)
def get_position_size_for_symbol(balance_usdt, symbol, step_size=0.00001, default_profile='medium'):
    profile = SYMBOL_RISKS.get(symbol.upper(), default_profile)
    multiplier = RISK_MULTIPLIERS.get(profile, RISK_MULTIPLIERS['medium'])
    usdt_to_risk = balance_usdt * multiplier
    price = get_precio_actual(symbol)
    if price is None or price <= 0:
        return None
    raw_amount = usdt_to_risk / price
    return ajustar_cantidad(raw_amount, step_size)

# Soporte / Resistencia simples: detectar n-top and n-bottom local extrema en lookback
def support_resistance_levels(df, lookback=50, n_extrema=3):
    """
    Devuelve listas de niveles de soporte y resistencia ordenadas.
    Usa máximos/mínimos locales simples y pivots.
    """
    closes = df['close']
    highs = df['high']
    lows = df['low']
    # usar rolling extrema
    local_max = highs.rolling(window=3, center=True).apply(lambda x: 1 if x[1] == max(x) else 0, raw=True)
    local_min = lows.rolling(window=3, center=True).apply(lambda x: 1 if x[1] == min(x) else 0, raw=True)
    res_levels = list(highs[local_max.fillna(0) == 1].tail(lookback).unique())
    sup_levels = list(lows[local_min.fillna(0) == 1].tail(lookback).unique())
    # fallback: usar top n global dentro lookback
    look = df.tail(lookback)
    top_n = sorted(look['high'].unique(), reverse=True)[:n_extrema]
    bot_n = sorted(look['low'].unique())[:n_extrema]
    res = sorted(set(res_levels + top_n), reverse=True)
    sup = sorted(set(sup_levels + bot_n))
    return sup[:n_extrema], res[:n_extrema]

# Heurística de compra/venta + hook a IA externa opcional
def ai_decision(symbol, df, rsi_threshold=40, support_margin_pct=0.5):
    """
    Devuelve (decision: bool, reason:str, confidence:float)
    - decision True => comprar
    - Default: regla simple: RSI < rsi_threshold AND precio dentro de margin% sobre un soporte
    - Puedes conectar aquí una llamada a un endpoint de IA (modelo), devolver su resultado combinado.
    """
    try:
        rsi = None
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
        price = df['close'].iloc[-1]
        sup_levels, res_levels = support_resistance_levels(df, lookback=60, n_extrema=3)
        near_support = False
        if sup_levels:
            nearest_sup = sup_levels[0]
            margin = nearest_sup * (support_margin_pct / 100)
            # si el precio está entre support y support + margin => near support
            if nearest_sup <= price <= nearest_sup + margin:
                near_support = True
        # regla base
        rule_buy = (rsi is not None and rsi < rsi_threshold and near_support)
        # placeholder IA: si existe ENDPOINT_IA en .env, llamarlo y combinar su output
        ia_endpoint = os.getenv('IA_DECISION_ENDPOINT')
        ia_confidence = 0.0
        ia_signal = None
        if ia_endpoint:
            try:
                payload = {
                    'symbol': symbol,
                    'close': float(price),
                    'rsi': float(rsi) if rsi is not None else None,
                    'supports': sup_levels,
                    'resistances': res_levels
                }
                resp = requests.post(ia_endpoint, json=payload, timeout=5)
                j = resp.json()
                ia_signal = j.get('buy')  # True/False
                ia_confidence = float(j.get('confidence', 0))
            except Exception:
                ia_signal = None
                ia_confidence = 0.0
        # combinación: si IA responde, seguir IA cuando confianza alta (>0.6), sino fallback rule
        if ia_signal is not None and ia_confidence >= 0.6:
            return bool(ia_signal), f"IA (conf {ia_confidence:.2f})", ia_confidence
        # si IA débil o no responde, usar regla
        if rule_buy:
            return True, f"Rule: RSI {rsi:.1f} < {rsi_threshold} and near support", 0.5
        return False, "No conditions met", 0.0
    except Exception as e:
        return False, f"Error in ai_decision: {e}", 0.0

# Example integration helper: decide and place buy
def try_entry(symbol, balance_usdt, step_size, max_positions_per_symbol=5):
    data = client.get_klines(symbol=symbol, interval='1h', limit=120)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','_','_','_','_','_','_'])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    # calculate rsi if TA available externally; here we just compute basic change RSI placeholder
    try:
        import ta
        df = df.copy()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    except Exception:
        df['rsi'] = df['close'].pct_change().rolling(14).mean()*100  # fallback crude
    decision, reason, conf = ai_decision(symbol, df)
    if not decision:
        return False, reason, conf
    # compute amount
    amount = get_position_size_for_symbol(balance_usdt, symbol, step_size=step_size)
    if amount is None or amount <= 0:
        return False, "No amount computed", 0.0
    return True, {'amount': amount, 'reason': reason, 'conf': conf}, conf

# Nota: aquí deberías integrar ejecutar_orden_con_confirmacion() y log_trade() de tu código actual.
# Ejemplo de uso (pseudocode):
# balance = get_balance()
# ok, info, conf = try_entry('BTCUSDT', balance, step_size)
# if ok: ejecutar_orden_con_confirmacion('buy', 'BTCUSDT', info['amount']) ; log_trade(...)

if __name__ == "__main__":
    # quick demo: imprimir tamaños para SYMBOLS desde .env
    SYMBOLS = [s.strip().upper() for s in os.getenv('SYMBOLS', 'BTCUSDT').split(',')]
    balance = 1000.0  # ejemplo: reemplazar con get_balance()
    step = float(os.getenv('STEP_SIZE', 0.00001))
    for s in SYMBOLS:
        amt = get_position_size_for_symbol(balance, s, step)
        print(f"{s}: profile={SYMBOL_RISKS.get(s,'default')} -> amount={amt}")