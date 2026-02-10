#!/usr/bin/env python3
"""Backtest de la estrategia (CSV o Binance Spot).

Permite ejecutar el backtest de la lógica principal con datos históricos:
- Fuente `csv`: lee OHLCV desde CSV (sin tocar red)
- Fuente `binance`: descarga velas desde Binance Spot (API pública) para el símbolo/intervalo/rango solicitado

Indicadores:
- RSI + StochRSI + EMA(ema_window) + ADX(adx_window) (lib `ta`)
Régimen:
- BULL/BEAR/LATERAL según EMA vs precio y ADX

CSV esperado (mínimo): timestamp, open, high, low, close
Opcional: volume
- timestamp puede ser ISO (2025-01-01 00:00:00) o epoch ms/seg.

Ejemplos:
    # 1) CSV (offline), opcional filtrar por rango
    python backtest_strategy.py \
        --csv data/BNBUSDT_1h.csv \
        --start 2024-01-01 --end 2024-03-01 \
        --initial-usdt 1000 --fee 0.001 --outdir backtest_out

    # 2) Binance (descarga velas y corre backtest)
    python backtest_strategy.py \
        --symbol BNBUSDT --interval 1h \
        --start 2024-01-01 --end 2024-03-01 \
        --cache-csv data_cache/BNBUSDT_1h_2024-01-01_2024-03-01.csv \
        --initial-usdt 1000 --fee 0.001 --outdir backtest_out
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import ta


# -------------------- Parámetros por régimen (copiados de bot_trading_v2_2.py) --------------------

BULL = {
    "RSI_THRESHOLD": 45,
    "TAKE_PROFIT_PCT": 5,
    "STOP_LOSS_PCT": 2,
    "TRAILING_TP_PCT": 1.2,
    "TRAILING_SL_PCT": 1.0,
    "BUY_COOLDOWN": 7200,      # 2h
    "POSITION_SIZE": 0.04,
}

BEAR = {
    "RSI_THRESHOLD": 30,
    "TAKE_PROFIT_PCT": 2,
    "STOP_LOSS_PCT": 1.2,
    "TRAILING_TP_PCT": 0.8,
    "TRAILING_SL_PCT": 0.6,
    "BUY_COOLDOWN": 21600,     # 6h
    "POSITION_SIZE": 0.015,
}

LATERAL = {
    "RSI_THRESHOLD": 40,
    "TAKE_PROFIT_PCT": 3,
    "STOP_LOSS_PCT": 1.5,
    "TRAILING_TP_PCT": 0.9,
    "TRAILING_SL_PCT": 0.8,
    "BUY_COOLDOWN": 14400,     # 4h
    "POSITION_SIZE": 0.03,
}

REGIME_PARAMS = {"BULL": BULL, "BEAR": BEAR, "LATERAL": LATERAL}


# -------------------- Helpers --------------------


def _coerce_timestamp(s: pd.Series) -> pd.Series:
    """Convierte columna timestamp a datetime.

    Soporta:
    - int/float epoch (ms o sec)
    - strings ISO
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    # si es numérico, inferimos ms vs sec
    if pd.api.types.is_numeric_dtype(s):
        # heurística: epoch ms suele ser > 10^12
        s2 = pd.to_numeric(s, errors="coerce")
        if s2.dropna().empty:
            return pd.to_datetime(s, errors="coerce")
        median = float(s2.dropna().median())
        unit = "ms" if median > 1e12 else "s"
        return pd.to_datetime(s2, unit=unit, errors="coerce")

    # strings
    return pd.to_datetime(s, errors="coerce", utc=False)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalizar nombres
    cols = {c.strip().lower(): c for c in df.columns}

    required = ["timestamp", "open", "high", "low", "close"]
    for r in required:
        if r not in cols:
            raise ValueError(
                f"CSV inválido: falta columna '{r}'. Columnas: {list(df.columns)}"
            )

    out = pd.DataFrame(
        {
            "timestamp": _coerce_timestamp(df[cols["timestamp"]]),
            "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        }
    )

    if "volume" in cols:
        out["volume"] = pd.to_numeric(df[cols["volume"]], errors="coerce")
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    out = out.reset_index(drop=True)
    return out


def cal_metrics(df: pd.DataFrame, rsi_window: int = 14, sma_short: int = 10, sma_long: int = 20) -> pd.DataFrame:
    ema_window = int(df.attrs.get("ema_window", 200))
    adx_window = int(df.attrs.get("adx_window", 14))

    df = df.copy()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_window).rsi()

    stochrsi = ta.momentum.StochRSIIndicator(df["close"], window=rsi_window, smooth1=3, smooth2=3)
    df["stochrsi_k"] = stochrsi.stochrsi_k()
    df["stochrsi_d"] = stochrsi.stochrsi_d()

    df["sma_short"] = ta.trend.SMAIndicator(df["close"], window=sma_short).sma_indicator()
    df["sma_long"] = ta.trend.SMAIndicator(df["close"], window=sma_long).sma_indicator()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # EMA / ADX (pueden ser NaN si no hay histórico suficiente)
    try:
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=ema_window).ema_indicator()
    except Exception:
        df["ema200"] = pd.NA

    try:
        adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=adx_window)
        df["adx"] = adx.adx()
    except Exception:
        df["adx"] = pd.NA

    return df


def detect_market_regime(row: pd.Series, *, adx_threshold: float = 25.0) -> str:
    """BULL/BEAR/LATERAL usando EMA200 y ADX."""
    try:
        price = row.get("close")
        ema200 = row.get("ema200")
        adx = row.get("adx")
        if pd.isna(price) or pd.isna(ema200) or pd.isna(adx):
            return "LATERAL"
        if price > ema200 and adx >= adx_threshold:
            return "BULL"
        if price < ema200 and adx >= adx_threshold:
            return "BEAR"
        return "LATERAL"
    except Exception:
        return "LATERAL"


def _parse_datetime_arg(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    # soporta epoch sec/ms como string
    if v.isdigit():
        num = int(v)
        # heurística ms vs sec
        if num > 1_000_000_000_000:
            ts = pd.to_datetime(num, unit="ms", errors="coerce")
        else:
            ts = pd.to_datetime(num, unit="s", errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Fecha inválida: {value}")
        t = pd.Timestamp(ts)
        return t.tz_localize(None) if t.tzinfo is not None else t

    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Fecha inválida: {value}")
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _to_millis(ts: pd.Timestamp) -> int:
    # Binance usa ms desde epoch
    return int(pd.Timestamp(ts).timestamp() * 1000)


def filter_df_by_period(df: pd.DataFrame, *, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if start is not None:
        out = out[out["timestamp"] >= start]
    if end is not None:
        out = out[out["timestamp"] < end]
    return out.reset_index(drop=True)


def fetch_binance_klines(
    *,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    base_url: str = "https://api.binance.com",
    limit: int = 1000,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """Descarga velas desde Binance Spot (endpoint /api/v3/klines).

    - Paginación por startTime/endTime (ms)
    - `end` es exclusivo (no incluye velas >= end)
    """
    if limit < 1 or limit > 1000:
        raise ValueError("limit debe estar entre 1 y 1000")
    if start is None:
        raise ValueError("start es requerido para descargar de Binance")

    url = base_url.rstrip("/") + "/api/v3/klines"
    start_ms = _to_millis(start)
    end_ms = _to_millis(end) if end is not None else None

    rows: list[list[Any]] = []
    next_start = start_ms
    while True:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            "startTime": next_start,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            break

        rows.extend(data)

        last_open_time_ms = int(data[-1][0])
        # Siguiente página: 1ms después de la última vela para evitar duplicados
        next_start = last_open_time_ms + 1

        # Si recibimos menos de limit, ya no hay más dentro del rango
        if len(data) < limit:
            break

        # Safety: si no avanza, cortamos
        if next_start <= last_open_time_ms:
            break

        # throttle para evitar rate limits
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(out["open_time"].astype("int64"), unit="ms", errors="coerce"),
            "open": pd.to_numeric(out["open"], errors="coerce"),
            "high": pd.to_numeric(out["high"], errors="coerce"),
            "low": pd.to_numeric(out["low"], errors="coerce"),
            "close": pd.to_numeric(out["close"], errors="coerce"),
            "volume": pd.to_numeric(out["volume"], errors="coerce"),
        }
    )

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    if end is not None:
        df = df[df["timestamp"] < end]
    return df.reset_index(drop=True)


def load_ohlcv_binance(
    *,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    base_url: str,
    cache_csv: str | None,
    limit: int,
) -> pd.DataFrame:
    if cache_csv and os.path.exists(cache_csv):
        df = load_ohlcv_csv(cache_csv)
        return filter_df_by_period(df, start=start, end=end)

    df = fetch_binance_klines(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        base_url=base_url,
        limit=limit,
    )

    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def pct_change_over_last_n_closes(df: pd.DataFrame, idx: int, n: int) -> float | None:
    """Retorno acumulado (close_{idx} vs open_{idx-n+1}) aproximado para DCA."""
    if idx - n + 1 < 0:
        return None
    first_open = float(df.loc[idx - n + 1, "open"])
    last_close = float(df.loc[idx, "close"])
    if first_open == 0:
        return None
    return ((last_close - first_open) / first_open) * 100.0


def _floor_to_step(qty: float, step_size: float) -> float:
    if step_size <= 0:
        return qty
    # floor simple; para backtest no necesitamos Decimal perfecto
    steps = int(qty / step_size)
    return steps * step_size


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    take_profit: float
    stop_loss: float
    regime: str
    rsi_threshold: float
    trailing_tp_pct: float
    trailing_sl_pct: float


@dataclass
class Trade:
    time: pd.Timestamp
    side: str
    price: float
    qty: float
    fee_paid: float
    pnl: float
    reason: str
    regime: str


def backtest(
    df: pd.DataFrame,
    *,
    initial_usdt: float,
    fee_rate: float,
    min_notional: float,
    max_positions_main: int,
    max_positions_dca: int,
    dca_drop_pct: float,
    dca_lookback: int,
    entry_at: str,
    step_size: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Ejecuta backtest.

    entry_at:
      - 'close': entra al cierre de la vela de señal
      - 'next_open': entra a la apertura de la siguiente vela (más conservador)
    """

    if entry_at not in ("close", "next_open"):
        raise ValueError("entry_at debe ser 'close' o 'next_open'")

    usdt = float(initial_usdt)
    positions: list[Position] = []
    trades: list[Trade] = []

    last_buy_ts: float | None = None

    equity_rows: list[dict[str, Any]] = []

    def mark_to_market(price: float) -> float:
        return usdt + sum(p.qty * price for p in positions)

    for i in range(len(df)):
        row = df.iloc[i]

        ts: pd.Timestamp = row["timestamp"]
        close = float(row["close"])

        # 1) gestión de posiciones con precio de cierre
        new_positions: list[Position] = []
        for p in positions:
            # trailing: si supera TP, "sube" TP/SL
            if close >= p.take_profit:
                p.take_profit = close * (1.0 + p.trailing_tp_pct / 100.0)
                p.stop_loss = close * (1.0 - p.trailing_sl_pct / 100.0)

            # salida por SL
            if close <= p.stop_loss:
                gross = p.qty * close
                fee_paid = gross * fee_rate
                proceeds = gross - fee_paid
                usdt_change = proceeds
                usdt_local_before = usdt
                usdt_local_after = usdt_local_before + usdt_change

                pnl = proceeds - (p.qty * p.entry_price)  # pnl sin fee de compra (se resta en buy)
                trades.append(
                    Trade(
                        time=ts,
                        side="sell",
                        price=close,
                        qty=p.qty,
                        fee_paid=fee_paid,
                        pnl=pnl,
                        reason="STOP_LOSS",
                        regime=p.regime,
                    )
                )
                usdt = usdt_local_after
                continue

            new_positions.append(p)

        positions = new_positions

        # 2) equity curve
        equity_rows.append({"timestamp": ts, "close": close, "usdt": usdt, "positions": len(positions), "equity": mark_to_market(close)})

        # 3) si indicadores no disponibles, skip
        rsi = row.get("rsi")
        k = row.get("stochrsi_k")
        d = row.get("stochrsi_d")
        if pd.isna(rsi) or pd.isna(k) or pd.isna(d):
            continue

        adx_threshold = float(df.attrs.get("adx_threshold", 25.0))
        regime = detect_market_regime(row, adx_threshold=adx_threshold)
        params = REGIME_PARAMS.get(regime, LATERAL)

        rsi_th = float(params["RSI_THRESHOLD"])
        tp_pct = float(params["TAKE_PROFIT_PCT"])
        sl_pct = float(params["STOP_LOSS_PCT"])
        trailing_tp = float(params["TRAILING_TP_PCT"])
        trailing_sl = float(params["TRAILING_SL_PCT"])
        cooldown = int(params["BUY_COOLDOWN"])
        pos_size = float(params["POSITION_SIZE"])

        # 4) cooldown
        can_buy = True
        now_s = ts.timestamp()
        if last_buy_ts is not None and (now_s - last_buy_ts) < cooldown:
            can_buy = False

        # 5) precio entrada
        if entry_at == "close":
            entry_price = close
            entry_ts = ts
        else:
            if i + 1 >= len(df):
                break
            next_row = df.iloc[i + 1]
            entry_price = float(next_row["open"])
            entry_ts = next_row["timestamp"]

        # 6) condición compra principal
        executed = False
        if can_buy and float(rsi) < rsi_th and float(k) > float(d) and usdt > min_notional and len(positions) < max_positions_main:
            capital = usdt * pos_size
            if capital >= min_notional and entry_price > 0:
                qty = capital / entry_price
                qty = _floor_to_step(qty, step_size)
                notional = qty * entry_price

                if notional >= min_notional and qty > 0:
                    buy_fee = notional * fee_rate
                    total_cost = notional + buy_fee
                    if total_cost <= usdt:
                        usdt -= total_cost
                        positions.append(
                            Position(
                                entry_time=entry_ts,
                                entry_price=entry_price,
                                qty=qty,
                                take_profit=entry_price * (1.0 + tp_pct / 100.0),
                                stop_loss=entry_price * (1.0 - sl_pct / 100.0),
                                regime=regime,
                                rsi_threshold=rsi_th,
                                trailing_tp_pct=trailing_tp,
                                trailing_sl_pct=trailing_sl,
                            )
                        )
                        trades.append(
                            Trade(
                                time=entry_ts,
                                side="buy",
                                price=entry_price,
                                qty=qty,
                                fee_paid=buy_fee,
                                pnl=0.0,
                                reason="BUY_MAIN",
                                regime=regime,
                            )
                        )
                        last_buy_ts = entry_ts.timestamp()
                        executed = True
                        if verbose:
                            print(f"[{entry_ts}] BUY_MAIN {regime} qty={qty:.6f} price={entry_price:.6f} usdt={usdt:.2f}")

        # 7) compra DCA por caída (aprox.)
        if (not executed) and can_buy and usdt > min_notional and len(positions) < max_positions_dca:
            movimiento = pct_change_over_last_n_closes(df, i, dca_lookback)
            if movimiento is not None and movimiento <= dca_drop_pct:
                capital = usdt * pos_size
                if capital >= min_notional and entry_price > 0:
                    qty = capital / entry_price
                    qty = _floor_to_step(qty, step_size)
                    notional = qty * entry_price

                    if notional >= min_notional and qty > 0:
                        buy_fee = notional * fee_rate
                        total_cost = notional + buy_fee
                        if total_cost <= usdt:
                            usdt -= total_cost
                            positions.append(
                                Position(
                                    entry_time=entry_ts,
                                    entry_price=entry_price,
                                    qty=qty,
                                    take_profit=entry_price * (1.0 + tp_pct / 100.0),
                                    stop_loss=entry_price * (1.0 - sl_pct / 100.0),
                                    regime=regime,
                                    rsi_threshold=rsi_th,
                                    trailing_tp_pct=trailing_tp,
                                    trailing_sl_pct=trailing_sl,
                                )
                            )
                            trades.append(
                                Trade(
                                    time=entry_ts,
                                    side="buy",
                                    price=entry_price,
                                    qty=qty,
                                    fee_paid=buy_fee,
                                    pnl=0.0,
                                    reason=f"BUY_DCA({movimiento:.2f}%)",
                                    regime=regime,
                                )
                            )
                            last_buy_ts = entry_ts.timestamp()
                            if verbose:
                                print(f"[{entry_ts}] BUY_DCA {regime} qty={qty:.6f} price={entry_price:.6f} mov={movimiento:.2f}% usdt={usdt:.2f}")

    # Cerrar posiciones al final (mark-to-market y salida)
    if len(df) > 0:
        last_ts = df.iloc[-1]["timestamp"]
        last_close = float(df.iloc[-1]["close"])
        for p in list(positions):
            gross = p.qty * last_close
            fee_paid = gross * fee_rate
            proceeds = gross - fee_paid
            pnl = proceeds - (p.qty * p.entry_price)
            trades.append(
                Trade(
                    time=last_ts,
                    side="sell",
                    price=last_close,
                    qty=p.qty,
                    fee_paid=fee_paid,
                    pnl=pnl,
                    reason="EOD_CLOSE",
                    regime=p.regime,
                )
            )
            usdt += proceeds
        positions = []

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_rows)

    # métricas
    if equity_df.empty:
        summary = {"initial": initial_usdt, "final": usdt, "return_pct": 0.0, "trades": 0}
        return trades_df, equity_df, summary

    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    max_dd = float(equity_df["drawdown"].min())

    sell_trades = trades_df[trades_df["side"] == "sell"] if not trades_df.empty else pd.DataFrame()
    wins = int((sell_trades["pnl"] > 0).sum()) if not sell_trades.empty else 0
    losses = int((sell_trades["pnl"] <= 0).sum()) if not sell_trades.empty else 0
    winrate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0

    final_equity = float(equity_df.iloc[-1]["equity"]) if len(equity_df) else float(usdt)
    ret_pct = ((final_equity - initial_usdt) / initial_usdt) * 100.0 if initial_usdt else 0.0

    summary = {
        "initial": float(initial_usdt),
        "final_equity": float(final_equity),
        "return_pct": float(ret_pct),
        "max_drawdown_pct": float(max_dd * 100.0),
        "num_trades": int(len(trades_df)),
        "num_sells": int(len(sell_trades)),
        "wins": wins,
        "losses": losses,
        "winrate": float(winrate * 100.0),
    }

    return trades_df, equity_df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest de estrategia (CSV o Binance Spot)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Ruta al CSV con OHLCV")
    src.add_argument("--symbol", help="Símbolo Binance (ej: BNBUSDT)")

    p.add_argument("--interval", default="1h", help="Intervalo de velas (solo Binance). Ej: 1m,5m,15m,1h,4h,1d")
    p.add_argument("--start", help="Inicio del período (YYYY-MM-DD o timestamp sec/ms). Recomendado")
    p.add_argument("--end", help="Fin del período (exclusivo). Si no se setea, usa 'ahora'")

    p.add_argument("--binance-base-url", default="https://api.binance.com", help="Base URL Spot API (default mainnet)")
    p.add_argument("--binance-limit", type=int, default=1000, help="Máx velas por request (1..1000)")
    p.add_argument("--cache-csv", help="Si se indica, guarda/lee las velas descargadas en este CSV")

    p.add_argument("--initial-usdt", type=float, default=1000.0, help="Capital inicial USDT")
    p.add_argument("--fee", type=float, default=0.001, help="Fee por trade (ej 0.001=0.1%)")
    p.add_argument("--min-notional", type=float, default=10.0, help="Mínimo notional para comprar")
    p.add_argument("--max-positions-main", type=int, default=5, help="Máximo posiciones en compra principal")
    p.add_argument("--max-positions-dca", type=int, default=9, help="Máximo posiciones en DCA")
    p.add_argument("--dca-drop-pct", type=float, default=-5.0, help="Umbral caída (%) para DCA")
    p.add_argument("--dca-lookback", type=int, default=5, help="Lookback velas para medir caída")
    p.add_argument("--entry-at", choices=["close", "next_open"], default="next_open", help="Precio de entrada")
    p.add_argument("--step-size", type=float, default=0.0, help="Redondeo qty (opcional) según step")

    p.add_argument("--rsi-window", type=int, default=14, help="Ventana RSI/StochRSI")
    p.add_argument("--sma-short", type=int, default=10, help="SMA corta")
    p.add_argument("--sma-long", type=int, default=20, help="SMA larga")
    p.add_argument("--ema-window", type=int, default=200, help="Ventana EMA para régimen (se guarda como columna ema200)")
    p.add_argument("--adx-window", type=int, default=14, help="Ventana ADX")
    p.add_argument("--adx-threshold", type=float, default=25.0, help="Umbral ADX para BULL/BEAR")

    p.add_argument(
        "--params-json",
        help=(
            "Ruta a JSON para sobreescribir REGIME_PARAMS (BULL/BEAR/LATERAL). "
            "Ej: {\"BULL\": {\"RSI_THRESHOLD\": 50, \"TAKE_PROFIT_PCT\": 6}}"
        ),
    )

    p.add_argument("--outdir", default="backtest_out", help="Directorio para outputs")
    p.add_argument("--verbose", action="store_true", help="Loggea cada operación")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    start = _parse_datetime_arg(args.start)
    end = _parse_datetime_arg(args.end)
    if end is None:
        end = pd.Timestamp.utcnow()

    if start is not None and end is not None and end <= start:
        raise ValueError("--end debe ser mayor que --start")

    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as rf:
            override = json.load(rf)
        if not isinstance(override, dict):
            raise ValueError("--params-json debe contener un objeto JSON")
        for k, v in override.items():
            if k in REGIME_PARAMS and isinstance(v, dict):
                REGIME_PARAMS[k].update(v)

    if args.csv:
        df = load_ohlcv_csv(args.csv)
        df = filter_df_by_period(df, start=start, end=end)
    else:
        if start is None:
            raise ValueError("Para --symbol (Binance) debes indicar --start")
        df = load_ohlcv_binance(
            symbol=args.symbol,
            interval=args.interval,
            start=start,
            end=end,
            base_url=args.binance_base_url,
            cache_csv=args.cache_csv,
            limit=args.binance_limit,
        )

    df.attrs["ema_window"] = int(args.ema_window)
    df.attrs["adx_window"] = int(args.adx_window)
    df.attrs["adx_threshold"] = float(args.adx_threshold)

    df = cal_metrics(df, rsi_window=args.rsi_window, sma_short=args.sma_short, sma_long=args.sma_long)

    trades_df, equity_df, summary = backtest(
        df,
        initial_usdt=args.initial_usdt,
        fee_rate=args.fee,
        min_notional=args.min_notional,
        max_positions_main=args.max_positions_main,
        max_positions_dca=args.max_positions_dca,
        dca_drop_pct=args.dca_drop_pct,
        dca_lookback=args.dca_lookback,
        entry_at=args.entry_at,
        step_size=args.step_size,
        verbose=args.verbose,
    )

    os.makedirs(args.outdir, exist_ok=True)
    trades_path = os.path.join(args.outdir, "trades.csv")
    equity_path = os.path.join(args.outdir, "equity_curve.csv")

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    print("=== Backtest summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"trades_csv: {trades_path}")
    print(f"equity_csv: {equity_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
