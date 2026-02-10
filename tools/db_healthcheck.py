#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_required(name: str) -> str:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _connect():
    import psycopg2

    host = _env_required("POSTGRES_HOST")
    port = _env_int("POSTGRES_PORT", 5432)
    dbname = _env_required("POSTGRES_DB")
    user = _env_required("POSTGRES_USER")
    password = _env_required("POSTGRES_PASSWORD")
    timeout_s = _env_int("POSTGRES_CONNECT_TIMEOUT", 5)

    dsn = (
        f"host={host} port={port} dbname={dbname} user={user} password={password} "
        f"connect_timeout={timeout_s} keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=5"
    )
    return psycopg2.connect(dsn)


def _ensure_schema_if_requested(conn) -> None:
    from pathlib import Path

    here = Path(__file__).resolve().parents[1]
    schema_sql = (here / "db" / "schema.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema_sql)


def _exists_schema(conn, schema: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)", (schema,))
        return bool(cur.fetchone()[0])


def _exists_table(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _table_columns(conn, schema: str, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {r[0] for r in cur.fetchall()}


def _indexes(conn, schema: str, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = %s AND tablename = %s
            """,
            (schema, table),
        )
        return {r[0] for r in cur.fetchall()}


def _counts(conn) -> dict:
    out = {}
    with conn.cursor() as cur:
        for t in ("open_positions", "trade_history", "equity_snapshots"):
            cur.execute(f"SELECT COUNT(*) FROM trading.{t}")
            out[t] = int(cur.fetchone()[0])
    return out


def _check_write_permissions(conn) -> None:
    # Write test without touching production tables:
    # - TEMP table lives only for this session
    # - everything is rolled back
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS __healthcheck_tmp (id int, ts timestamptz)")
        cur.execute("INSERT INTO __healthcheck_tmp(id, ts) VALUES (1, now())")
        cur.execute("SELECT COUNT(*) FROM __healthcheck_tmp")
        n = int(cur.fetchone()[0])
        if n != 1:
            raise RuntimeError("TEMP write/read check failed")


def run_checks(*, ensure_schema: bool, check_write: bool) -> tuple[list[CheckResult], dict]:
    results: list[CheckResult] = []
    meta: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not _env_bool("USE_DATABASE", False):
        raise RuntimeError("USE_DATABASE must be true for this bot")

    t0 = time.time()
    conn = None
    try:
        conn = _connect()
        conn.autocommit = False
        results.append(CheckResult(True, "connect", "Connected"))

        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            _ = cur.fetchone()[0]
            cur.execute("SHOW server_version")
            meta["server_version"] = cur.fetchone()[0]

        results.append(CheckResult(True, "select_1", "OK"))

        if ensure_schema:
            try:
                _ensure_schema_if_requested(conn)
                conn.commit()
                results.append(CheckResult(True, "ensure_schema", "Applied db/schema.sql"))
            except Exception as e:
                conn.rollback()
                results.append(CheckResult(False, "ensure_schema", str(e)))

        schema_ok = _exists_schema(conn, "trading")
        results.append(CheckResult(schema_ok, "schema_trading", "Exists" if schema_ok else "Missing"))

        required_tables = {
            "open_positions": {
                "id",
                "symbol",
                "buy_price",
                "amount",
                "take_profit",
                "stop_loss",
                "regime",
                "atr_entry",
                "tp_atr_mult",
                "sl_atr_mult",
                "trailing_sl_atr_mult",
                "trailing_active",
                "max_price",
                "opened_at",
                "updated_at",
            },
            "trade_history": {
                "id",
                "symbol",
                "side",
                "reason",
                "buy_price",
                "sell_price",
                "amount",
                "realized_pnl",
                "realized_pnl_pct",
                "volatility",
                "rsi",
                "stochrsi_k",
                "stochrsi_d",
                "regime",
                "executed_at",
            },
            "equity_snapshots": {
                "id",
                "timestamp",
                "equity_total",
                "usdt_balance",
                "positions_value",
                "positions_json",
                "created_at",
            },
        }

        for table, required_cols in required_tables.items():
            exists = _exists_table(conn, "trading", table)
            results.append(CheckResult(exists, f"table_trading.{table}", "Exists" if exists else "Missing"))
            if not exists:
                continue
            cols = _table_columns(conn, "trading", table)
            missing = sorted(required_cols - cols)
            results.append(
                CheckResult(
                    len(missing) == 0,
                    f"columns_trading.{table}",
                    "OK" if not missing else f"Missing columns: {', '.join(missing)}",
                )
            )

        # Index checks (best-effort; names are defined in db/schema.sql)
        idx_open = _indexes(conn, "trading", "open_positions")
        results.append(
            CheckResult(
                {"idx_open_positions_symbol", "idx_open_positions_opened_at"}.issubset(idx_open),
                "indexes_open_positions",
                f"Found: {', '.join(sorted(idx_open))}" if idx_open else "No indexes found",
            )
        )

        idx_trade = _indexes(conn, "trading", "trade_history")
        results.append(
            CheckResult(
                {
                    "idx_trade_history_symbol_executed_at",
                    "idx_trade_history_executed_at",
                    "idx_trade_history_reason",
                }.issubset(idx_trade),
                "indexes_trade_history",
                f"Found: {', '.join(sorted(idx_trade))}" if idx_trade else "No indexes found",
            )
        )

        idx_equity = _indexes(conn, "trading", "equity_snapshots")
        results.append(
            CheckResult(
                {"ux_equity_snapshots_timestamp", "idx_equity_snapshots_timestamp"}.issubset(idx_equity),
                "indexes_equity_snapshots",
                f"Found: {', '.join(sorted(idx_equity))}" if idx_equity else "No indexes found",
            )
        )

        if check_write:
            try:
                _check_write_permissions(conn)
                conn.rollback()  # ensure we leave no traces
                results.append(CheckResult(True, "write_temp_table", "OK (rolled back)"))
            except Exception as e:
                conn.rollback()
                results.append(CheckResult(False, "write_temp_table", str(e)))

        # Basic analytics counters
        try:
            meta["counts"] = _counts(conn)
            results.append(CheckResult(True, "counts", json.dumps(meta["counts"])))
        except Exception as e:
            conn.rollback()
            results.append(CheckResult(False, "counts", str(e)))

        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return results, meta

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL healthcheck for the trading bot")
    parser.add_argument("--env-file", default=None, help="Optional .env file path")
    parser.add_argument("--ensure-schema", action="store_true", help="Apply db/schema.sql before checks")
    parser.add_argument("--no-write-check", action="store_true", help="Skip TEMP write permission test")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    try:
        results, meta = run_checks(ensure_schema=bool(args.ensure_schema), check_write=(not args.no_write_check))
    except Exception as e:
        if args.as_json:
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        else:
            print(f"ERROR: {e}")
        return 2

    ok_all = all(r.ok for r in results)

    if args.as_json:
        payload = {
            "ok": ok_all,
            "meta": meta,
            "checks": [r.__dict__ for r in results],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"OK={ok_all} duration_ms={meta.get('duration_ms')} server={meta.get('server_version')}")
        for r in results:
            status = "OK" if r.ok else "FAIL"
            detail = f" — {r.detail}" if r.detail else ""
            print(f"[{status}] {r.name}{detail}")

    return 0 if ok_all else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
