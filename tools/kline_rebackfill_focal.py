from __future__ import annotations

"""Focal re-backfill for missing klines in Postgres.

This tool is meant to repair small gaps detected by the continuity warning:
  "Gap in fetched klines: prev=... curr=... expected_step_ms=..."

It will:
- compute the missing open_time(s) between prev and curr for the given interval
- for each missing open_time, request Binance klines in a tiny [t, t+step) window
- insert any returned closed candle(s) into Postgres (ON CONFLICT DO NOTHING)
- re-audit the same gap to confirm what's still missing

Usage examples:
  python tools/kline_rebackfill_focal.py --symbol BTCUSDT --interval 1h \
      --prev-ms 1618880400000 --curr-ms 1618891200000

  python tools/kline_rebackfill_focal.py --symbol BTCUSDT --interval 1h \
      --prev-ms 1614992400000 --curr-ms 1614999600000

Notes:
- Uses public Binance klines; API keys are optional.
- Requires Postgres env vars (POSTGRES_*) in repo-root .env or environment.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from binance.client import Client

from db.connection import init_db_from_env
from repositories.market_klines_repo import MarketKlinesRepository
from services.market_klines_service import interval_to_milliseconds


def _load_env() -> None:
    try:
        repo_root_env = Path(__file__).resolve().parents[1] / ".env"
        if repo_root_env.exists():
            load_dotenv(dotenv_path=repo_root_env, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        pass


def _ms_to_dt_naive_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _dt_to_ms_utc_naive(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


@dataclass(frozen=True)
class Gap:
    prev_open: datetime
    curr_open: datetime
    step_ms: int

    def missing_opens(self) -> list[datetime]:
        out: list[datetime] = []
        step = timedelta(milliseconds=int(self.step_ms))
        t = self.prev_open + step
        while t < self.curr_open:
            out.append(t)
            t += step
        return out


def _parse_binance_kline(raw: list) -> dict:
    # see services.market_klines_service._parse_binance_kline
    open_time_ms = int(raw[0])
    close_time_ms = int(raw[6])

    def _ms_to_dt(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)

    return {
        "open_time": _ms_to_dt(open_time_ms),
        "close_time": _ms_to_dt(close_time_ms),
        "open": raw[1],
        "high": raw[2],
        "low": raw[3],
        "close": raw[4],
        "volume": raw[5],
        "quote_volume": raw[7],
        "trades": int(raw[8]),
    }


def _audit_missing_in_db(
    *,
    repo: MarketKlinesRepository,
    symbol: str,
    interval: str,
    missing_opens: list[datetime],
    step_ms: int,
) -> list[datetime]:
    if not missing_opens:
        return []

    start = min(missing_opens)
    end = max(missing_opens) + timedelta(milliseconds=int(step_ms))
    rows = repo.read_range(symbol=symbol, interval=interval, start_time=start, end_time=end, limit=None) or []
    present = {r.get("open_time") for r in rows if r.get("open_time") is not None}

    # DB stores naive UTC datetimes; keep same
    out = []
    for dt in missing_opens:
        if dt not in present:
            out.append(dt)
    return out


def main(argv: list[str] | None = None) -> int:
    _load_env()

    ap = argparse.ArgumentParser(description="Focal re-backfill missing klines for a known prev/curr gap.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--prev-ms", type=int, required=True)
    ap.add_argument("--curr-ms", type=int, required=True)
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep seconds between Binance calls")
    ap.add_argument("--max-retries", type=int, default=2, help="Retries per missing candle")
    ap.add_argument(
        "--probe-on-miss",
        action="store_true",
        help="When a candle can't be fetched, probe Binance and print the next returned open_time(s).",
    )
    args = ap.parse_args(argv)

    sym = str(args.symbol).upper().strip()
    itv = str(args.interval).strip()

    step_ms = int(interval_to_milliseconds(itv))
    prev_dt = _ms_to_dt_naive_utc(int(args.prev_ms))
    curr_dt = _ms_to_dt_naive_utc(int(args.curr_ms))
    if curr_dt <= prev_dt:
        raise SystemExit("--curr-ms must be greater than --prev-ms")

    gap = Gap(prev_open=prev_dt, curr_open=curr_dt, step_ms=step_ms)
    missing = gap.missing_opens()

    print(f"Symbol={sym} Interval={itv}")
    print(f"Gap prev={prev_dt.isoformat()} curr={curr_dt.isoformat()} step_ms={step_ms}")
    if not missing:
        print("No missing opens expected for this gap (diff matches one step).")
        return 0

    print(f"Missing opens expected: {len(missing)}")
    for dt in missing:
        print(f"  {dt.isoformat()}")

    db = init_db_from_env(require_use_database=False)
    repo = MarketKlinesRepository(db)

    # If some are already present, skip them
    missing_in_db = _audit_missing_in_db(repo=repo, symbol=sym, interval=itv, missing_opens=missing, step_ms=step_ms)
    if not missing_in_db:
        print("All expected missing opens are already present in DB. Nothing to do.")
        return 0

    print(f"Missing opens NOT in DB: {len(missing_in_db)}")

    # Binance client (public klines; keys optional)
    client_data = Client(None, None, testnet=False)

    inserted_total = 0
    failed: list[datetime] = []

    for dt in missing_in_db:
        start_ms = _dt_to_ms_utc_naive(dt)
        end_ms = start_ms + step_ms

        ok = False
        for attempt in range(1, int(args.max_retries) + 2):
            try:
                raw = client_data.get_klines(
                    symbol=sym,
                    interval=itv,
                    startTime=int(start_ms),
                    endTime=int(end_ms),
                    limit=2,
                )

                # Keep only CLOSED candles (close_time <= now)
                now_ms = int(time.time() * 1000)
                raw = [r for r in (raw or []) if int(r[6]) <= now_ms]

                parsed = [_parse_binance_kline(r) for r in raw]

                # Only insert the candle matching this exact open_time
                parsed = [p for p in parsed if p.get("open_time") == dt]

                if not parsed:
                    ok = False
                else:
                    repo.insert_many(symbol=sym, interval=itv, rows=parsed)
                    inserted_total += len(parsed)
                    ok = True

                if float(args.sleep) > 0:
                    time.sleep(float(args.sleep))

                if ok:
                    print(f"[OK] inserted {dt.isoformat()}")
                    break
                else:
                    print(f"[MISS] Binance returned no closed candle for {dt.isoformat()} (attempt {attempt})")
                    if args.probe_on_miss:
                        try:
                            probe = client_data.get_klines(
                                symbol=sym,
                                interval=itv,
                                startTime=int(start_ms),
                                limit=5,
                            )
                            opens = [int(r[0]) for r in (probe or [])][:5]
                            if opens:
                                print(
                                    "       probe_open_times="
                                    + ",".join(str(x) for x in opens)
                                    + f" (first={_ms_to_dt_naive_utc(opens[0]).isoformat()})"
                                )
                            else:
                                print("       probe_open_times=(none)")
                        except Exception as pe:
                            print(f"       probe_error={pe}")
            except Exception as e:
                print(f"[ERR] {dt.isoformat()} attempt {attempt}: {e}")

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

        if not ok:
            failed.append(dt)

    print(f"Inserted attempted rows: {inserted_total}")

    # Re-audit
    still_missing = _audit_missing_in_db(repo=repo, symbol=sym, interval=itv, missing_opens=missing, step_ms=step_ms)
    print(f"Still missing in DB after re-backfill: {len(still_missing)}")
    for dt in still_missing:
        print(f"  {dt.isoformat()}")

    if failed:
        print(f"Failed to fetch/insert: {len(failed)}")
        for dt in failed:
            print(f"  {dt.isoformat()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
