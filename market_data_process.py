"""Market Data Process (NO TRADING)

Responsabilidad única:
- Descargar klines OHLCV desde Binance MAINNET (client_data)
- Backfill inicial + sincronización incremental
- Validar continuidad temporal (best-effort)
- Persistir en PostgreSQL (trading.market_klines)

Reglas:
- NO ejecutar órdenes
- NO consultar precios en tiempo real
- NO usar client_trade

Ejecución típica:
- Corre como servicio separado (systemd/docker/k8s CronJob/Deployment)
- Mantiene Postgres siempre "caliente" para que el Trading Bot lea solo desde DB
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from datetime import datetime

from binance.client import Client
from dotenv import load_dotenv

from db.connection import init_db_from_env
from db.schema import ensure_market_schema
from services.market_klines_service import sync_klines_to_postgres


logger = logging.getLogger(__name__)


def _env_list(name: str, default_csv: str) -> list[str]:
    raw = os.getenv(name, default_csv) or default_csv
    return [x.strip().upper() for x in str(raw).split(",") if x.strip()]


def _env_list_raw(name: str, default_csv: str) -> list[str]:
    raw = os.getenv(name, default_csv) or default_csv
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _env_datetime(name: str) -> datetime | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    # Support common ISO8601 forms, including trailing 'Z'
    s = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        logger.warning("Invalid %s=%s (expected ISO8601). Ignoring.", name, raw)
        return None


_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True


def build_client_data() -> Any:
    # MAINNET data client. Keys are optional for public klines endpoint.
    data_key = (os.getenv("BINANCE_DATA_API_KEY") or "").strip() or None
    data_secret = (os.getenv("BINANCE_DATA_API_SECRET") or "").strip() or None
    return Client(data_key, data_secret, testnet=False)


def main() -> int:
    logging.basicConfig(
        level=(os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load env vars from `.env` colocated with this script (robust when launched from other cwd).
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except Exception:
        # Non-fatal: env can still be provided via the shell/container.
        pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    symbols = _env_list("MARKET_DATA_SYMBOLS", os.getenv("SYMBOLS", "BTCUSDT"))
    intervals = _env_list_raw("MARKET_DATA_INTERVALS", os.getenv("TIMEFRAME", "1h"))

    backfill_limit = _env_int("KLINES_BACKFILL_LIMIT", 1000)
    max_pages = _env_int("KLINES_MAX_PAGES", 20)
    overlap_candles = _env_int("KLINES_OVERLAP_CANDLES", 2)

    # Optional history controls
    history_lookback = (os.getenv("KLINES_HISTORY_LOOKBACK") or "").strip() or None
    start_time = _env_datetime("KLINES_START_TIME")
    end_time = _env_datetime("KLINES_END_TIME")

    sync_every_s = _env_float("MARKET_DATA_SYNC_EVERY_SECONDS", 30.0)

    logger.info("Market Data Process starting symbols=%s intervals=%s", symbols, intervals)

    db = init_db_from_env()
    # Market Data Process must NOT touch trading tables (open_positions/trade_history/etc.)
    ensure_market_schema(db)

    client_data = build_client_data()

    # Simple loop: sync each symbol/interval. Idempotent due to PK + ON CONFLICT DO NOTHING.
    while not _STOP:
        started = time.time()
        for sym in symbols:
            for itv in intervals:
                if _STOP:
                    break
                try:
                    res = sync_klines_to_postgres(
                        db=db,
                        client_data=client_data,
                        symbol=sym,
                        interval=itv,
                        backfill_limit=backfill_limit,
                        history_lookback=history_lookback,
                        start_time=start_time,
                        end_time=end_time,
                        max_pages=max_pages,
                        overlap_candles=overlap_candles,
                    )
                    logger.info(
                        "Synced %s %s fetched=%s inserted_attempted=%s last_open_time=%s",
                        res.symbol,
                        res.interval,
                        res.fetched,
                        res.inserted,
                        res.last_open_time,
                    )
                except Exception as e:
                    logger.exception("Sync failed for %s %s: %s", sym, itv, e)

        elapsed = time.time() - started
        sleep_s = max(0.0, float(sync_every_s) - elapsed)
        time.sleep(sleep_s)

    logger.warning("Market Data Process stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
