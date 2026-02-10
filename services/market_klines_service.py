import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from db.connection import PostgresDatabase
from repositories.market_klines_repo import MarketKlinesRepository


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KlineSyncResult:
    symbol: str
    interval: str
    fetched: int
    inserted: int
    last_open_time: Optional[datetime]


def interval_to_milliseconds(interval: str) -> int:
    s = str(interval).strip().lower()
    if not s:
        raise ValueError("interval vacío")

    unit = s[-1]
    try:
        value = int(s[:-1])
    except Exception as e:
        raise ValueError(f"interval inválido: {interval}") from e

    if value <= 0:
        raise ValueError(f"interval inválido: {interval}")

    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    if unit == "w":
        return value * 7 * 24 * 60 * 60_000

    raise ValueError(f"interval no soportado: {interval}")


def _ms_to_dt_utc_naive(ms: int) -> datetime:
    # Store as naive UTC (TIMESTAMP without time zone)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _parse_binance_kline(raw: list[Any]) -> dict:
    # python-binance spot klines format:
    # [
    #  0 Open time,
    #  1 Open,
    #  2 High,
    #  3 Low,
    #  4 Close,
    #  5 Volume,
    #  6 Close time,
    #  7 Quote asset volume,
    #  8 Number of trades,
    #  9 Taker buy base asset volume,
    #  10 Taker buy quote asset volume,
    #  11 Ignore
    # ]
    open_time_ms = int(raw[0])
    close_time_ms = int(raw[6])
    return {
        "open_time": _ms_to_dt_utc_naive(open_time_ms),
        "close_time": _ms_to_dt_utc_naive(close_time_ms),
        "open": raw[1],
        "high": raw[2],
        "low": raw[3],
        "close": raw[4],
        "volume": raw[5],
        "quote_volume": raw[7],
        "trades": int(raw[8]),
    }


def sync_klines_to_postgres(
    *,
    db: PostgresDatabase,
    client_data: Any,
    symbol: str,
    interval: str,
    backfill_limit: int = 1000,
    max_pages: int = 20,
    overlap_candles: int = 2,
) -> KlineSyncResult:
    """Incremental MAINNET kline ingest -> PostgreSQL.

    - Reads last stored open_time.
    - If empty: backfills last `backfill_limit` klines.
    - Else: fetches only new klines from last_open_time - overlap.
    - Inserts via UPSERT (DO NOTHING).

    Notes:
    - Uses naive UTC timestamps in DB.
    - `overlap_candles` avoids gaps due to late/partial candles; duplicates are ignored.
    """

    repo = MarketKlinesRepository(db)
    sym = str(symbol).upper().strip()
    itv = str(interval).strip()

    last_open = repo.last_open_time(symbol=sym, interval=itv)
    interval_ms = interval_to_milliseconds(itv)

    fetched_total = 0
    inserted_total = 0

    # Decide startTime
    start_time_ms: Optional[int]
    if last_open is None:
        start_time_ms = None
    else:
        # last_open is naive UTC
        last_open_utc = last_open.replace(tzinfo=timezone.utc)
        last_open_ms = int(last_open_utc.timestamp() * 1000)
        start_time_ms = max(0, last_open_ms - (overlap_candles * interval_ms))

    page = 0
    while page < max_pages:
        page += 1

        if last_open is None and start_time_ms is None:
            raw = client_data.get_klines(symbol=sym, interval=itv, limit=int(backfill_limit))
        else:
            raw = client_data.get_klines(symbol=sym, interval=itv, startTime=int(start_time_ms), limit=1000)

        if not raw:
            break

        # Only store CLOSED candles. Binance may return the currently-forming candle which will change.
        now_ms = int(time.time() * 1000)
        raw = [r for r in raw if int(r[6]) <= now_ms]
        if not raw:
            break

        parsed = [_parse_binance_kline(r) for r in raw]
        fetched_total += len(parsed)

        # Continuity check inside fetched chunk (best-effort)
        for i in range(1, len(raw)):
            prev_open_ms = int(raw[i - 1][0])
            curr_open_ms = int(raw[i][0])
            if curr_open_ms != prev_open_ms + interval_ms:
                logger.warning(
                    "[%s %s] Gap in fetched klines: prev=%s curr=%s expected_step_ms=%s",
                    sym,
                    itv,
                    prev_open_ms,
                    curr_open_ms,
                    interval_ms,
                )
                break

        # Insert
        try:
            repo.insert_many(symbol=sym, interval=itv, rows=parsed)
            inserted_total += len(parsed)  # attempted; duplicates ignored by PK
        except Exception:
            logger.exception("[%s %s] Failed inserting klines batch", sym, itv)
            raise

        # If this was a backfill request (no last_open), we stop after one page.
        if last_open is None:
            break

        # Next page startTime = last open_time of this response + interval
        last_open_ms_in_page = int(raw[-1][0])
        next_ms = last_open_ms_in_page + interval_ms
        if start_time_ms is not None and next_ms <= start_time_ms:
            break
        start_time_ms = next_ms

        # Heuristic stop: if we got less than max, we're caught up.
        if len(raw) < 1000:
            break

        # Avoid hammering API
        time.sleep(0.15)

    new_last_open = repo.last_open_time(symbol=sym, interval=itv)
    return KlineSyncResult(
        symbol=sym,
        interval=itv,
        fetched=fetched_total,
        inserted=inserted_total,
        last_open_time=new_last_open,
    )


def read_klines_df(
    *,
    db: PostgresDatabase,
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """Reads latest klines from PostgreSQL into a Pandas DataFrame.

    IMPORTANT: This function is *pure storage read*.
    - It does NOT call Binance.
    - Syncing/backfill MUST be done by the Market Data Process.
    """

    repo = MarketKlinesRepository(db)
    sym = str(symbol).upper().strip()
    itv = str(interval).strip()

    rows = repo.read_latest(symbol=sym, interval=itv, limit=int(limit))
    if not rows:
        return pd.DataFrame(columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
        ])

    df = pd.DataFrame(rows)

    # Normalize columns to match existing bot expectations
    df = df.rename(columns={"open_time": "timestamp"})

    # Data types
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

    # Ensure sorted ascending by timestamp
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df
