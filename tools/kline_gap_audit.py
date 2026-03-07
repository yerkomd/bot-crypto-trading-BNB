from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Iterable

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from db.connection import init_db_from_env
from services.market_klines_service import interval_to_milliseconds


def _load_env() -> None:
    """Load env vars from repo-root .env if present (same convention as bot/backtesting)."""
    try:
        repo_root_env = Path(__file__).resolve().parents[1] / ".env"
        if repo_root_env.exists():
            load_dotenv(dotenv_path=repo_root_env, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        pass


def _parse_dt(value: str) -> datetime:
    s = str(value).strip()
    if not s:
        raise ValueError("Empty datetime")
    # Accept YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).replace(tzinfo=None)
    # Accept ISO8601, including trailing Z
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        # assume UTC
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


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

    @property
    def missing_count(self) -> int:
        diff_ms = _dt_to_ms_utc_naive(self.curr_open) - _dt_to_ms_utc_naive(self.prev_open)
        if self.step_ms <= 0:
            return 0
        n = int(diff_ms // self.step_ms) - 1
        return max(0, n)

    def missing_opens(self) -> list[datetime]:
        out: list[datetime] = []
        step = timedelta(milliseconds=int(self.step_ms))
        t = self.prev_open + step
        while t < self.curr_open:
            out.append(t)
            t += step
        return out


def _fetch_open_times(
    *,
    db,
    schema: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    sym = str(symbol).upper().strip()
    itv = str(interval).strip()
    sch = str(schema).strip() or "trading"

    def _q(cur):
        cur.execute(
            f"""
            SELECT open_time
            FROM {sch}.market_klines
            WHERE symbol = %s
              AND interval = %s
              AND open_time >= %s
              AND open_time < %s
            ORDER BY open_time ASC
            """,
            (sym, itv, start, end),
        )
        rows = cur.fetchall()
        return [r["open_time"] for r in rows]

    rows = db.run(_q, retries=2, swallow=False)
    out = []
    for dt in rows:
        if isinstance(dt, datetime):
            out.append(dt.replace(tzinfo=None))
    return out


def _iter_gaps(open_times: Iterable[datetime], step_ms: int) -> list[Gap]:
    ots = list(open_times)
    gaps: list[Gap] = []
    for i in range(1, len(ots)):
        prev = ots[i - 1]
        curr = ots[i]
        diff_ms = _dt_to_ms_utc_naive(curr) - _dt_to_ms_utc_naive(prev)
        if diff_ms != int(step_ms):
            gaps.append(Gap(prev_open=prev, curr_open=curr, step_ms=int(step_ms)))
    return gaps


def _print_gap(g: Gap, *, max_missing: int) -> None:
    print(
        f"GAP prev={g.prev_open.isoformat()} curr={g.curr_open.isoformat()} "
        f"missing={g.missing_count} step_ms={g.step_ms}"
    )
    missing = g.missing_opens()
    if missing:
        for dt in missing[: max_missing if max_missing > 0 else len(missing)]:
            print(f"  missing_open={dt.isoformat()}")
        if max_missing > 0 and len(missing) > max_missing:
            print(f"  ... ({len(missing) - max_missing} more)")


def main(argv: list[str] | None = None) -> int:
    _load_env()

    ap = argparse.ArgumentParser(
        description=(
            "Audit kline continuity in Postgres (trading.market_klines) for a symbol/interval. "
            "Supports checking a specific prev_ms/curr_ms gap, or scanning a time window."
        )
    )
    ap.add_argument("--schema", default="trading", help="DB schema (default: trading)")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    ap.add_argument("--interval", required=True, help="Interval, e.g. 1h")

    ap.add_argument("--prev-ms", type=int, default=None, help="Prev open_time in epoch ms")
    ap.add_argument("--curr-ms", type=int, default=None, help="Curr open_time in epoch ms")

    ap.add_argument("--start", type=str, default=None, help="Scan start (ISO8601 or YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="Scan end (ISO8601 or YYYY-MM-DD). Exclusive.")

    ap.add_argument("--max-gaps", type=int, default=25, help="Max gaps to print in scan mode")
    ap.add_argument("--max-missing", type=int, default=20, help="Max missing opens to print per gap")

    args = ap.parse_args(argv)

    step_ms = int(interval_to_milliseconds(str(args.interval)))

    if (args.prev_ms is None) ^ (args.curr_ms is None):
        raise SystemExit("Provide both --prev-ms and --curr-ms, or neither")

    db = init_db_from_env(require_use_database=False)

    # Mode A: check a specific gap (prev_ms -> curr_ms)
    if args.prev_ms is not None and args.curr_ms is not None:
        prev_dt = _ms_to_dt_naive_utc(int(args.prev_ms))
        curr_dt = _ms_to_dt_naive_utc(int(args.curr_ms))
        if curr_dt <= prev_dt:
            raise SystemExit("curr must be > prev")

        g = Gap(prev_open=prev_dt, curr_open=curr_dt, step_ms=step_ms)
        print(f"Symbol={str(args.symbol).upper().strip()} Interval={str(args.interval).strip()} Schema={args.schema}")
        _print_gap(g, max_missing=int(args.max_missing))

        expected = g.missing_opens()
        if not expected:
            print("No missing opens expected for this prev/curr (diff matches step).")
            return 0

        # Fetch all opens in (prev, curr) and compare
        rows = _fetch_open_times(
            db=db,
            schema=str(args.schema),
            symbol=str(args.symbol),
            interval=str(args.interval),
            start=prev_dt + timedelta(milliseconds=step_ms),
            end=curr_dt,
        )
        present = set(rows)

        missing_in_db = [dt for dt in expected if dt not in present]
        present_in_db = [dt for dt in expected if dt in present]

        print(f"DB rows found inside gap: {len(rows)}")
        print(f"Expected missing opens: {len(expected)}")
        print(f"Present in DB: {len(present_in_db)}")
        print(f"Missing in DB: {len(missing_in_db)}")

        if missing_in_db:
            print("Missing opens not found in DB:")
            for dt in missing_in_db:
                print(f"  {dt.isoformat()}")

        return 0

    # Mode B: scan a range
    if not args.start or not args.end:
        raise SystemExit("In scan mode you must provide --start and --end")

    start_dt = _parse_dt(str(args.start))
    end_dt = _parse_dt(str(args.end))
    if end_dt <= start_dt:
        raise SystemExit("--end must be > --start")

    rows = _fetch_open_times(
        db=db,
        schema=str(args.schema),
        symbol=str(args.symbol),
        interval=str(args.interval),
        start=start_dt,
        end=end_dt,
    )

    print(f"Symbol={str(args.symbol).upper().strip()} Interval={str(args.interval).strip()} Schema={args.schema}")
    print(f"Range [{start_dt.isoformat()}, {end_dt.isoformat()}): rows={len(rows)}")

    if len(rows) < 2:
        print("Not enough rows to detect gaps.")
        return 0

    gaps = _iter_gaps(rows, step_ms)
    print(f"Gaps detected: {len(gaps)}")

    for g in gaps[: max(0, int(args.max_gaps))]:
        _print_gap(g, max_missing=int(args.max_missing))

    if int(args.max_gaps) > 0 and len(gaps) > int(args.max_gaps):
        print(f"... ({len(gaps) - int(args.max_gaps)} more gaps not shown)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
