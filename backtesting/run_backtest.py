from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from backtesting import BacktestEngine, BacktestConfig
from backtesting.strategies import BotV3_1StrategyAdapter, BotV2_2StrategyAdapter, BotV4StrategyAdapter, BotV5StrategyAdapter
from db.connection import init_db_from_env
from services.market_klines_service import read_klines_df, read_klines_df_range


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None

    # epoch seconds/ms
    if v.isdigit():
        n = int(v)
        if n > 1_000_000_000_000:
            ts = pd.to_datetime(n, unit="ms", errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(n, unit="s", errors="coerce", utc=True)
        if pd.isna(ts):
            raise ValueError(f"Invalid datetime: {value}")
        return ts.to_pydatetime().replace(tzinfo=None)

    ts = pd.to_datetime(v, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime: {value}")
    return ts.to_pydatetime().replace(tzinfo=None)


def _parse_symbols(s: str) -> list[str]:
    syms = [x.strip().upper() for x in str(s).split(",") if x.strip()]
    if not syms:
        raise ValueError("--symbols is required")
    return syms


def _load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    required = ["timestamp", "open", "high", "low", "close"]
    for r in required:
        if r not in cols:
            raise ValueError(f"CSV missing column '{r}'. Have: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[cols["timestamp"]], errors="coerce", utc=True).dt.tz_convert(None),
            "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols["volume"]], errors="coerce") if "volume" in cols else 0.0,
        }
    )
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtesting.run_backtest",
        description=(
            "Backtest candle-by-candle using production bot strategy adapters (no lookahead). "
            "Select with --strategy: v2_2, v3_1, v4, v5."
        ),
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. CLI flags override config values.",
    )
    p.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to a YAML config file (requires PyYAML). CLI flags override config values.",
    )

    p.add_argument(
        "--source",
        choices=["postgres", "csv"],
        default="postgres",
        help="Data source for OHLCV bars. postgres reads trading.market_klines; csv reads a local file.",
    )

    p.add_argument(
        "--strategy",
        choices=["v2_2", "v3_1", "v4", "v5"],
        default="v3_1",
        help="Which strategy adapter to use: v2_2, v3_1, v4, or v5.",
    )
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--interval", type=str, default=os.getenv("TIMEFRAME", "1h"), help="Kline interval, e.g. 1h")

    p.add_argument("--start", type=str, default=None, help="Start datetime (inclusive). ISO or epoch sec/ms.")
    p.add_argument("--end", type=str, default=None, help="End datetime (exclusive). ISO or epoch sec/ms.")
    p.add_argument(
        "--lookback-bars",
        type=int,
        default=None,
        help="If start/end not provided (postgres only), read last N bars.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max bars to read (range query only). Optional safety cap.",
    )

    p.add_argument("--csv", type=str, default=None, help="CSV file path when --source=csv")

    # Engine config
    p.add_argument("--initial-cash", type=float, default=1000.0)
    p.add_argument("--fee-rate", type=float, default=0.001, help="Fee rate per side (default 0.001 = 0.1 percent).")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in bps (e.g. 5 = 0.05 percent).")
    p.add_argument("--max-positions-per-symbol", type=int, default=5)
    p.add_argument("--min-notional", type=float, default=10.0)
    p.add_argument("--hard-cooldown-s", type=float, default=0.0)

    # Strategy overrides (without editing bot_trading_v3_1.py)
    p.add_argument("--rsi-min", type=float, default=None, help="Override RSI_CONFIRM_MIN for entry.")
    p.add_argument("--rsi-max", type=float, default=None, help="Override RSI_CONFIRM_MAX for entry.")
    p.add_argument(
        "--regime-params-json",
        type=str,
        default=None,
        help="JSON override for REGIME_PARAMS. Example: '{" 
        "\"BULL\": {\"BUY_COOLDOWN\": 7200, \"POSITION_SIZE\": 0.08}," 
        "\"LATERAL\": {\"BUY_COOLDOWN\": 14400, \"POSITION_SIZE\": 0.03}," 
        "\"BEAR\": {\"BUY_COOLDOWN\": 21600, \"POSITION_SIZE\": 0.0}}'",
    )
    p.add_argument(
        "--atr-multipliers-json",
        type=str,
        default=None,
        help="JSON override for ATR_MULTIPLIERS. Example: '{" 
        "\"BULL\": {\"tp\": 2.5, \"sl\": 1.5, \"trailing_sl\": 1.2}," 
        "\"LATERAL\": {\"tp\": 2.0, \"sl\": 1.2, \"trailing_sl\": 1.0}," 
        "\"BEAR\": null}'",
    )
    p.add_argument(
        "--v5-threshold-override",
        type=float,
        default=None,
        help="Override operating_threshold from v5 ML artifact (backtesting only).",
    )
    p.add_argument(
        "--v5-model-path",
        type=str,
        default=None,
        help="Path to v5 ML artifact joblib file. Defaults to artifacts/model_momentum_v1.joblib.",
    )

    p.add_argument("--output-dir", type=str, default="backtest_out", help="Directory to write equity_curve.csv & trades.csv")

    return p


def _load_config_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON in config file: {p}") from e
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object/dict")
    return data


def _load_config_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config-yaml. Install it with: pip install PyYAML"
        ) from e
    try:
        data = yaml.safe_load(raw)
    except Exception as e:
        raise ValueError(f"Invalid YAML in config file: {p}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping/object")
    return data


def _cfg_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    except Exception:
        pass
    return default


def main(argv: list[str] | None = None) -> int:
    # 1) Pre-parse only --config (so we can load defaults)
    pre = argparse.ArgumentParser(add_help=False)
    # Load env vars from repo-root `.env` (same convention as the trading bot).
    # This makes Postgres backtests work without manual `export POSTGRES_*`.
    try:
        repo_root_env = Path(__file__).resolve().parents[1] / ".env"
        if repo_root_env.exists():
            load_dotenv(dotenv_path=repo_root_env, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        # dotenv is optional at runtime; env vars may already be set.
        pass
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("--config-yaml", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.config and pre_args.config_yaml:
        raise SystemExit("Use only one of --config or --config-yaml")

    cfg = {}
    if pre_args.config_yaml:
        cfg = _load_config_yaml(pre_args.config_yaml)
    else:
        cfg = _load_config_json(pre_args.config)

    # 2) Parse full CLI (CLI values override config)
    args = build_arg_parser().parse_args(argv)

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    cli_has_strategy = "--strategy" in argv_list
    cli_has_source = "--source" in argv_list

    # Apply config fallbacks for common fields
    if not cli_has_strategy:
        args.strategy = str(_cfg_get(cfg, "strategy", args.strategy))
    if not cli_has_source:
        args.source = str(_cfg_get(cfg, "source", args.source))
    if not args.symbols:
        args.symbols = _cfg_get(cfg, "symbols", "")
    if not args.interval:
        args.interval = _cfg_get(cfg, "interval", os.getenv("TIMEFRAME", "1h"))
    if args.start is None:
        args.start = _cfg_get(cfg, "start", None)
    if args.end is None:
        args.end = _cfg_get(cfg, "end", None)
    if args.lookback_bars is None:
        args.lookback_bars = _cfg_get(cfg, "lookback_bars", None)
    if args.limit is None:
        args.limit = _cfg_get(cfg, "limit", None)
    if args.csv is None:
        args.csv = _cfg_get(cfg, "csv", None)
    if args.output_dir == "backtest_out":
        args.output_dir = _cfg_get(cfg, "output_dir", args.output_dir)

    # Engine config fallbacks
    if args.initial_cash == 1000.0:
        args.initial_cash = float(_cfg_get(cfg, "initial_cash", args.initial_cash))
    if args.fee_rate == 0.001:
        args.fee_rate = float(_cfg_get(cfg, "fee_rate", args.fee_rate))
    if args.slippage_bps == 0.0:
        args.slippage_bps = float(_cfg_get(cfg, "slippage_bps", args.slippage_bps))
    if args.max_positions_per_symbol == 5:
        args.max_positions_per_symbol = int(_cfg_get(cfg, "max_positions_per_symbol", args.max_positions_per_symbol))
    if args.min_notional == 10.0:
        args.min_notional = float(_cfg_get(cfg, "min_notional", args.min_notional))
    if args.hard_cooldown_s == 0.0:
        args.hard_cooldown_s = float(_cfg_get(cfg, "hard_cooldown_s", args.hard_cooldown_s))

    # Strategy overrides fallbacks
    if args.rsi_min is None:
        args.rsi_min = _cfg_get(cfg, "rsi_min", None)
    if args.rsi_max is None:
        args.rsi_max = _cfg_get(cfg, "rsi_max", None)
    if args.v5_threshold_override is None:
        args.v5_threshold_override = _cfg_get(cfg, "v5_threshold_override", None)
    if args.v5_model_path is None:
        args.v5_model_path = _cfg_get(cfg, "v5_model_path", None)

    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv is required when --source=csv")
        symbols = _parse_symbols(args.symbols) if args.symbols else ["CSV"]
        if len(symbols) != 1:
            raise SystemExit("--source=csv supports exactly one symbol (use --symbols ONE)")
        symbol = symbols[0]
        df = _load_csv(args.csv)
        start = _parse_dt(args.start)
        end = _parse_dt(args.end)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] < pd.Timestamp(end)]
        data_by_symbol = {symbol: df.reset_index(drop=True)}

    else:
        symbols = _parse_symbols(args.symbols)
        start = _parse_dt(args.start)
        end = _parse_dt(args.end)

        db = init_db_from_env(require_use_database=False)
        data_by_symbol: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            if start is not None and end is not None:
                df = read_klines_df_range(
                    db=db,
                    symbol=sym,
                    interval=str(args.interval),
                    start=start,
                    end=end,
                    limit=args.limit,
                )
            else:
                if args.lookback_bars is None:
                    raise SystemExit("For --source=postgres you must provide --start and --end, or --lookback-bars")
                df = read_klines_df(db=db, symbol=sym, interval=str(args.interval), limit=int(args.lookback_bars))

            n = 0 if df is None else int(len(df))
            if df is None or df.empty:
                print(f"[DATA] {sym}: 0 bars (empty)")
            else:
                ts0 = pd.to_datetime(df["timestamp"].iloc[0]).to_pydatetime()
                ts1 = pd.to_datetime(df["timestamp"].iloc[-1]).to_pydatetime()
                print(f"[DATA] {sym}: {n} bars {ts0} -> {ts1}")
            data_by_symbol[sym] = df

        if not any(df is not None and not df.empty for df in data_by_symbol.values()):
            raise SystemExit(
                "No market data loaded from Postgres. "
                "Check: POSTGRES_* env vars, that trading.market_klines contains the requested symbols/interval, "
                "and that your --start/--end window matches stored timestamps."
            )

    regime_params = None
    if args.regime_params_json:
        regime_params = json.loads(args.regime_params_json)
    else:
        regime_params = _cfg_get(cfg, "regime_params", None)

    atr_multipliers = None
    if args.atr_multipliers_json:
        atr_multipliers = json.loads(args.atr_multipliers_json)
    else:
        atr_multipliers = _cfg_get(cfg, "atr_multipliers", None)

    strategy_name = str(args.strategy or "v3_1").strip().lower()
    if strategy_name == "v3_1":
        print("[STRATEGY] Using: v3_1 (BotV3_1StrategyAdapter -> bot_trading_v3_1)")
        strategy = BotV3_1StrategyAdapter(
            rsi_confirm_min=args.rsi_min,
            rsi_confirm_max=args.rsi_max,
            regime_params=regime_params,
            atr_multipliers=atr_multipliers,
        )
    elif strategy_name == "v2_2":
        # v2_2 ignores RSI_CONFIRM_MIN/MAX & ATR multipliers; it uses regime params (RSI_THRESHOLD, TP/SL pct, trailing pct, etc).
        print("[STRATEGY] Using: v2_2 (BotV2_2StrategyAdapter -> bot_trading_v2_2)")
        strategy = BotV2_2StrategyAdapter(regime_params=regime_params)
    elif strategy_name == "v4":
        # v4 is a simple EMA200 + breakout strategy with fixed ATR SL/trailing (no regime params).
        print("[STRATEGY] Using: v4 (BotV4StrategyAdapter -> bot_trading_v4)")
        strategy = BotV4StrategyAdapter()
    elif strategy_name == "v5":
        print("[STRATEGY] Using: v5 (BotV5StrategyAdapter -> estrategia_v5)")
        strategy = BotV5StrategyAdapter(
            threshold_override=args.v5_threshold_override,
            model_path=args.v5_model_path,
        )
    else:
        raise SystemExit(f"Unknown --strategy: {args.strategy}")

    cfg = BacktestConfig(
        initial_cash=float(args.initial_cash),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        max_positions_per_symbol=int(args.max_positions_per_symbol),
        min_notional=float(args.min_notional),
        hard_cooldown_s=float(args.hard_cooldown_s),
    )

    engine = BacktestEngine(strategy=strategy, config=cfg)
    out_dir = Path(args.output_dir)
    res = engine.run(data_by_symbol=data_by_symbol, output_dir=out_dir)

    print("Backtest completed")
    print(f"Equity curve: {res.equity_curve_path}")
    print(f"Trades: {res.trades_path}")
    print("Metrics:")
    print(json.dumps(asdict(res.metrics), indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
