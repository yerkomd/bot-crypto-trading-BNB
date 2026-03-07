-- Idempotent schema for Market Data Process (NO trading)
-- Creates ONLY the OHLCV time-series table used as historical source of truth.

CREATE SCHEMA IF NOT EXISTS trading;
SET search_path TO trading, public;

-- Market klines (OHLCV time-series)
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
    -- Volumes can be very large on higher timeframes (e.g. 1w). Use a wider numeric.
    volume NUMERIC(28,8),
    quote_volume NUMERIC(28,8),
    trades INTEGER,
    created_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_market_klines_symbol_interval_open_time_desc
    ON trading.market_klines (symbol, interval, open_time DESC);
