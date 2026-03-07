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


def lookback_to_milliseconds(lookback: str) -> int:
    """Parses a simple lookback string into milliseconds.

    Supported units: m, h, d, w.
    Examples: "90d", "12h", "30m", "2w".
    """

    s = str(lookback).strip().lower()
    if not s:
        raise ValueError("lookback vacío")

    unit = s[-1]
    try:
        value = int(s[:-1])
    except Exception as e:
        raise ValueError(f"lookback inválido: {lookback}") from e

    if value <= 0:
        raise ValueError(f"lookback inválido: {lookback}")

    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    if unit == "w":
        return value * 7 * 24 * 60 * 60_000

    raise ValueError(f"lookback no soportado: {lookback}")


def _ms_to_dt_utc_naive(ms: int) -> datetime:
    # Store as naive UTC (TIMESTAMP without time zone)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _dt_to_ms_utc(dt: datetime) -> int:
    """Converts datetime to unix ms assuming UTC.

    - If dt is naive: assumed UTC.
    - If dt is aware: converted to UTC.
    """

    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


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
    history_lookback: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    max_pages: int = 20,
    overlap_candles: int = 2,
) -> KlineSyncResult:
    """Incremental MAINNET kline ingest -> PostgreSQL.

    - Reads last stored open_time.
    - If empty and no explicit range is provided: backfills last `backfill_limit` klines.
    - If empty and `history_lookback` is provided (e.g. "90d"): backfills that time range.
    - If `start_time`/`end_time` is provided: fetches within that explicit range.
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

    # Decide startTime/endTime
    now_ms = int(time.time() * 1000)
    end_time_ms: Optional[int] = _dt_to_ms_utc(end_time) if end_time is not None else None

    explicit_start_ms: Optional[int] = _dt_to_ms_utc(start_time) if start_time is not None else None
    if explicit_start_ms is None and history_lookback:
        try:
            lookback_ms = lookback_to_milliseconds(history_lookback)
            explicit_start_ms = max(0, now_ms - lookback_ms)
        except Exception:
            logger.exception("Invalid history_lookback=%s", history_lookback)
            raise

    start_time_ms: Optional[int]
    if last_open is None:
        start_time_ms = explicit_start_ms
    else:
        # last_open is naive UTC
        last_open_ms = _dt_to_ms_utc(last_open)
        incremental_ms = max(0, last_open_ms - (overlap_candles * interval_ms))
        # If an explicit start was given, allow going further back to fill gaps.
        start_time_ms = explicit_start_ms if explicit_start_ms is not None else incremental_ms

    page = 0
    initial_db_empty = last_open is None
    single_page_backfill = initial_db_empty and start_time_ms is None and end_time_ms is None and not history_lookback and start_time is None and end_time is None
    while page < max_pages:
        page += 1

        if single_page_backfill:
            raw = client_data.get_klines(symbol=sym, interval=itv, limit=int(backfill_limit))
        else:
            kwargs = {"symbol": sym, "interval": itv, "limit": 1000}
            if start_time_ms is not None:
                kwargs["startTime"] = int(start_time_ms)
            if end_time_ms is not None:
                kwargs["endTime"] = int(end_time_ms)
            raw = client_data.get_klines(**kwargs)

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

        # If this was a simple backfill request (DB empty and no range), we stop after one page.
        if single_page_backfill:
            break

        # Next page startTime = last open_time of this response + interval
        last_open_ms_in_page = int(raw[-1][0])
        next_ms = last_open_ms_in_page + interval_ms
        if end_time_ms is not None and next_ms > end_time_ms:
            break
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


def read_klines_df_range(
    *,
    db: PostgresDatabase,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    limit: int | None = None,
) -> pd.DataFrame:
    """Reads klines from PostgreSQL for a specific time range into a Pandas DataFrame.

    IMPORTANT: This function is *pure storage read*.
    - It does NOT call Binance.
    - It does NOT backfill/sync.

    Range is [start, end) in UTC (naive datetimes; matches DB TIMESTAMP storage).
    """

    repo = MarketKlinesRepository(db)
    sym = str(symbol).upper().strip()
    itv = str(interval).strip()

    rows = repo.read_range(symbol=sym, interval=itv, start_time=start, end_time=end, limit=limit)
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
    df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df
