-- Run this as a PostgreSQL superuser (postgres) OR as the DB owner role (e.g. db_admin).
-- Purpose: allow the app role to create/use schema `trading` and create tables inside it.
--
-- Assumptions (edit if needed):
--   - Database name: bot_trading
--   - App role/user: bot_trading_user
--   - Optional read-only role: analytics_user

-- 0) Connection permissions
GRANT CONNECT ON DATABASE bot_trading TO bot_trading_user;

-- 1) IMPORTANT: allow schema creation in this database.
-- Without this, you'll get: "permission denied for database bot_trading" when creating schema/tables.
GRANT CREATE ON DATABASE bot_trading TO bot_trading_user;

-- 2) Create schema `trading` owned by the app role (run while connected to db=bot_trading)
CREATE SCHEMA IF NOT EXISTS trading AUTHORIZATION bot_trading_user;

-- 3) Ensure app role can use/create objects in that schema
GRANT USAGE, CREATE ON SCHEMA trading TO bot_trading_user;

-- Optional: read-only analytics role
-- GRANT USAGE ON SCHEMA trading TO analytics_user;

-- 4) Default privileges for future objects created BY bot_trading_user in schema trading.
-- This is mainly useful for granting access to other roles (e.g., analytics_user).
ALTER DEFAULT PRIVILEGES FOR ROLE bot_trading_user IN SCHEMA trading
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO bot_trading_user;

ALTER DEFAULT PRIVILEGES FOR ROLE bot_trading_user IN SCHEMA trading
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO bot_trading_user;

-- Optional: grant read-only by default on new tables to analytics_user
-- ALTER DEFAULT PRIVILEGES FOR ROLE bot_trading_user IN SCHEMA trading
-- GRANT SELECT ON TABLES TO analytics_user;

-- ALTER DEFAULT PRIVILEGES FOR ROLE bot_trading_user IN SCHEMA trading
-- GRANT USAGE, SELECT ON SEQUENCES TO analytics_user;
