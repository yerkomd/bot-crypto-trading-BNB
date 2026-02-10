-- Idempotent schema for the trading bot
-- Schema: trading

CREATE SCHEMA IF NOT EXISTS trading;

-- Optional: keep unqualified names resolved as expected
SET search_path TO trading, public;

-- 1) Open positions (live state)
CREATE TABLE IF NOT EXISTS trading.open_positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    buy_price NUMERIC(18,8) NOT NULL,
    amount NUMERIC(18,8) NOT NULL,
    take_profit NUMERIC(18,8),
    stop_loss NUMERIC(18,8),
    regime VARCHAR(10),
    atr_entry NUMERIC(18,8),
    tp_atr_mult NUMERIC(10,4),
    sl_atr_mult NUMERIC(10,4),
    trailing_sl_atr_mult NUMERIC(10,4),
    trailing_active BOOLEAN DEFAULT false,
    max_price NUMERIC(18,8),
    opened_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_open_positions_symbol ON trading.open_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_open_positions_opened_at ON trading.open_positions(opened_at);

-- 2) Trade history (FACT table)
CREATE TABLE IF NOT EXISTS trading.trade_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    reason VARCHAR(20),
    buy_price NUMERIC(18,8),
    sell_price NUMERIC(18,8),
    amount NUMERIC(18,8),
    realized_pnl NUMERIC(18,8),
    realized_pnl_pct NUMERIC(10,4),
    volatility NUMERIC(10,4),
    rsi NUMERIC(10,4),
    stochrsi_k NUMERIC(10,4),
    stochrsi_d NUMERIC(10,4),
    regime VARCHAR(10),
    executed_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_executed_at ON trading.trade_history(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_trade_history_executed_at ON trading.trade_history(executed_at);
CREATE INDEX IF NOT EXISTS idx_trade_history_reason ON trading.trade_history(reason);

-- 3) Equity snapshots (time-series)
CREATE TABLE IF NOT EXISTS trading.equity_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    equity_total NUMERIC(18,8) NOT NULL,
    usdt_balance NUMERIC(18,8) NOT NULL,
    positions_value NUMERIC(18,8) NOT NULL,
    positions_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_equity_snapshots_timestamp ON trading.equity_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_timestamp ON trading.equity_snapshots(timestamp);

-- 4) Market klines (OHLCV time-series)
-- NOTE: timestamps are stored as UTC in TIMESTAMP (without timezone) for portability.
CREATE TABLE IF NOT EXISTS trading.market_klines (
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(5) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    open NUMERIC(18,8),
    high NUMERIC(18,8),
    low NUMERIC(18,8),
    close NUMERIC(18,8),
    volume NUMERIC(18,8),
    quote_volume NUMERIC(18,8),
    trades INTEGER,
    created_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_market_klines_symbol_interval_open_time_desc
    ON trading.market_klines (symbol, interval, open_time DESC);
