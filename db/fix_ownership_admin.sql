-- Fix ownership for existing tables/indexes so the app role can manage schema migrations.
--
-- Run this as a PostgreSQL superuser (postgres) or the current OWNER of the objects.
--
-- Why:
--   Errors like: "must be owner of table open_positions" happen when the app connects
--   with a role that is NOT the owner, and `db/schema.sql` tries to create indexes.
--
-- How to use:
--   1) Edit app_role below to match your app user (usually POSTGRES_USER from .env).
--   2) Execute this file in the target database.

DO $$
DECLARE
    app_role text := 'trader';
BEGIN
    -- Schema
    EXECUTE format('ALTER SCHEMA trading OWNER TO %I', app_role);

    -- Tables
    EXECUTE format('ALTER TABLE IF EXISTS trading.open_positions OWNER TO %I', app_role);
    EXECUTE format('ALTER TABLE IF EXISTS trading.trade_history OWNER TO %I', app_role);
    EXECUTE format('ALTER TABLE IF EXISTS trading.equity_snapshots OWNER TO %I', app_role);
    EXECUTE format('ALTER TABLE IF EXISTS trading.market_klines OWNER TO %I', app_role);

    -- Sequences (default names for BIGSERIAL columns)
    EXECUTE format('ALTER SEQUENCE IF EXISTS trading.open_positions_id_seq OWNER TO %I', app_role);
    EXECUTE format('ALTER SEQUENCE IF EXISTS trading.trade_history_id_seq OWNER TO %I', app_role);
    EXECUTE format('ALTER SEQUENCE IF EXISTS trading.equity_snapshots_id_seq OWNER TO %I', app_role);

    -- Basic privileges (optional hardening)
    EXECUTE format('GRANT USAGE, CREATE ON SCHEMA trading TO %I', app_role);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA trading TO %I', app_role);
    EXECUTE format('GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA trading TO %I', app_role);
END $$;
