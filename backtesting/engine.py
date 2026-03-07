from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import pandas as pd

from .metrics import compute_metrics, Metrics
from .bt_types import Bar, Position, Trade, Strategy, StrategyContext, EntrySignal


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1000.0
    fee_rate: float = 0.001  # 0.1%
    slippage_bps: float = 0.0

    max_positions_per_symbol: int = 5
    min_notional: float = 10.0

    # Cooldown is handled by the strategy (if it wants), but engine can enforce a minimum too.
    hard_cooldown_s: float = 0.0

    # Conservative assumption: if both TP and SL are inside the same candle, assume SL happens first.
    conservative_fill_order: bool = True


@dataclass(frozen=True)
class BacktestResult:
    metrics: Metrics
    equity_curve_path: Path
    trades_path: Path


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _slippage_frac(slippage_bps: float) -> float:
    try:
        bps = float(slippage_bps)
    except Exception:
        bps = 0.0
    return max(0.0, bps / 10000.0)


class BacktestEngine:
    def __init__(self, *, strategy: Strategy, config: Optional[BacktestConfig] = None):
        self._strategy = strategy
        self._cfg = config or BacktestConfig()

    def run(
        self,
        *,
        data_by_symbol: dict[str, pd.DataFrame],
        output_dir: str | Path = ".",
    ) -> BacktestResult:
        """Run a multi-symbol backtest.

        Expected DataFrame schema per symbol: columns include
        `timestamp`, `open`, `high`, `low`, `close`.

        No-lookahead policy:
        - signals are computed from bar i-1 (closed candle)
        - entries execute at bar i open
        - exits (TP/SL) are checked using bar i high/low
        """

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prepared: dict[str, pd.DataFrame] = {}
        for sym, df in data_by_symbol.items():
            if df is None or df.empty:
                continue
            prepared[sym] = self._strategy.prepare_indicators(symbol=sym, df=df)

        if not prepared:
            equity_curve_path = out_dir / "equity_curve.csv"
            trades_path = out_dir / "trades.csv"
            eq_columns = ["timestamp", "cash", "positions_value", "equity", "open_positions", "drawdown"]
            tr_columns = [
                "symbol",
                "entry_time",
                "exit_time",
                "qty",
                "entry_price",
                "exit_price",
                "reason",
                "pnl",
                "pnl_pct",
                "fees_paid",
                "duration_bars",
            ]
            pd.DataFrame([], columns=eq_columns).to_csv(equity_curve_path, index=False)
            pd.DataFrame([], columns=tr_columns).to_csv(trades_path, index=False)
            metrics = compute_metrics(timestamps=[], equity=[], trade_pnls=[], trade_entry_ts=[], trade_exit_ts=[])
            return BacktestResult(metrics=metrics, equity_curve_path=equity_curve_path, trades_path=trades_path)

        # Build global timeline (union of timestamps)
        timelines: dict[str, list[datetime]] = {}
        for sym, df in prepared.items():
            ts = pd.to_datetime(df["timestamp"]).tolist()
            timelines[sym] = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in ts]

        global_ts_sorted = sorted({t for ts in timelines.values() for t in ts})

        # State
        cash = float(self._cfg.initial_cash)
        positions: list[Position] = []
        last_entry_time_by_symbol: dict[str, datetime] = {}
        last_close_by_symbol: dict[str, float] = {}

        trades: list[Trade] = []
        equity_rows: list[dict[str, Any]] = []

        slip = _slippage_frac(self._cfg.slippage_bps)

        # For each symbol, keep an index pointer into its bars
        idx_by_symbol: dict[str, int] = {sym: 0 for sym in prepared.keys()}

        for t in global_ts_sorted:
            # Process each symbol that has a bar at time t
            for sym, df in prepared.items():
                i = idx_by_symbol[sym]
                if i >= len(df):
                    continue
                bar_ts = df["timestamp"].iloc[i]
                bar_ts = pd.to_datetime(bar_ts).to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
                if bar_ts != t:
                    continue

                # Bar i is the bar we can simulate (entry at open, exit within high/low)
                bar = Bar(
                    timestamp=t,
                    open=float(df["open"].iloc[i]),
                    high=float(df["high"].iloc[i]),
                    low=float(df["low"].iloc[i]),
                    close=float(df["close"].iloc[i]),
                    volume=_to_float(df["volume"].iloc[i]) if "volume" in df.columns else None,
                )
                last_close_by_symbol[sym] = float(bar.close)

                # Entry signals use bar i-1 (closed candle)
                if i >= 1:
                    signal_row = df.iloc[i - 1]
                    indicators_row = {k: signal_row.get(k) for k in df.columns}
                    regime = str(indicators_row.get("regime") or "LATERAL")

                    # Strategy may compute regime from indicators row; we allow it to overwrite
                    # by returning a computed regime in meta.
                    # Context equity uses latest closes known.
                    positions_value = 0.0
                    for p in positions:
                        px = last_close_by_symbol.get(p.symbol)
                        if px is None:
                            continue
                        positions_value += float(p.qty) * float(px)
                    equity = float(cash + positions_value)

                    ctx = StrategyContext(
                        symbol=sym,
                        i=i,
                        timestamp=t,
                        indicators=indicators_row,
                        regime=str(regime),
                        cash=float(cash),
                        equity=float(equity),
                        positions_open_symbol=sum(1 for p in positions if p.symbol == sym),
                        last_entry_time=last_entry_time_by_symbol.get(sym),
                    )
                    sig: EntrySignal = self._strategy.generate_entry(ctx)

                    if sig.should_enter:
                        # Enforce max positions per symbol + min notional
                        if ctx.positions_open_symbol >= int(self._cfg.max_positions_per_symbol):
                            sig = EntrySignal(False, sig.position_size_frac, sig.meta)
                        if cash <= float(self._cfg.min_notional):
                            sig = EntrySignal(False, sig.position_size_frac, sig.meta)

                    # Hard cooldown enforcement (optional)
                    if sig.should_enter and self._cfg.hard_cooldown_s > 0:
                        last_t = last_entry_time_by_symbol.get(sym)
                        if last_t is not None and (t - last_t).total_seconds() < float(self._cfg.hard_cooldown_s):
                            sig = EntrySignal(False, sig.position_size_frac, sig.meta)

                    if sig.should_enter and sig.position_size_frac > 0:
                        capital = float(cash) * float(sig.position_size_frac)
                        if capital >= float(self._cfg.min_notional):
                            entry_price = float(bar.open) * (1.0 + slip)
                            qty = capital / entry_price if entry_price > 0 else 0.0
                            notional = qty * entry_price
                            fee_buy = notional * float(self._cfg.fee_rate)

                            if qty > 0 and (notional + fee_buy) <= cash:
                                take_profit, stop_loss, extra = self._strategy.compute_risk_levels(
                                    symbol=sym,
                                    regime=str(ctx.regime),
                                    buy_price=float(entry_price),
                                    indicators_row=indicators_row,
                                )
                                p = Position(
                                    symbol=sym,
                                    qty=float(qty),
                                    entry_time=t,
                                    entry_index=int(i),
                                    entry_price=float(entry_price),
                                    regime=str(ctx.regime),
                                    take_profit=float(take_profit),
                                    stop_loss=float(stop_loss),
                                    tp_initial=float(extra.get("tp_initial", take_profit)),
                                    trailing_active=bool(extra.get("trailing_active", False)),
                                    max_price=float(extra.get("max_price", entry_price)),
                                    entry_fee=float(fee_buy),
                                    atr_entry=_to_float(extra.get("atr_entry")),
                                    tp_atr_mult=_to_float(extra.get("tp_atr_mult")),
                                    sl_atr_mult=_to_float(extra.get("sl_atr_mult")),
                                    trailing_sl_atr_mult=_to_float(extra.get("trailing_sl_atr_mult")),
                                )

                                cash -= float(notional + fee_buy)
                                positions.append(p)
                                last_entry_time_by_symbol[sym] = t

                # Exits for existing positions (that were open before bar)
                for p in list(positions):
                    if p.symbol != sym:
                        continue
                    exited, new_cash, trade = self._check_exit_in_bar(
                        bar=bar,
                        pos=p,
                        fee_rate=float(self._cfg.fee_rate),
                        slip=float(slip),
                        conservative=bool(self._cfg.conservative_fill_order),
                        bar_index=i,
                    )
                    if exited and trade is not None:
                        cash = float(cash + new_cash)
                        positions.remove(p)
                        trades.append(trade)

                # Trailing updates at end of bar (no lookahead)
                for p in positions:
                    if p.symbol != sym:
                        continue
                    # Use indicators at bar close i
                    row_i = df.iloc[i]
                    indicators_now = {k: row_i.get(k) for k in df.columns}
                    self._strategy.update_trailing(symbol=sym, position=p, bar=bar, indicators_row=indicators_now)

                idx_by_symbol[sym] = i + 1

            # After processing all symbols at time t, snapshot equity
            positions_value = 0.0
            for p in positions:
                px = last_close_by_symbol.get(p.symbol)
                if px is None:
                    continue
                positions_value += float(p.qty) * float(px)
            equity = float(cash + positions_value)

            # Drawdown from peak
            if equity_rows:
                peak = max(float(r["equity"]) for r in equity_rows)
            else:
                peak = equity
            dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

            equity_rows.append(
                {
                    "timestamp": t,
                    "cash": float(cash),
                    "positions_value": float(positions_value),
                    "equity": float(equity),
                    "drawdown": float(dd),
                    "open_positions": int(len(positions)),
                }
            )

        # Liquidate remaining positions at last close
        if equity_rows:
            last_t = equity_rows[-1]["timestamp"]
        else:
            last_t = global_ts_sorted[-1]

        last_index_by_symbol = {sym: max(0, int(idx_by_symbol.get(sym, 0)) - 1) for sym in prepared.keys()}

        for p in list(positions):
            px = last_close_by_symbol.get(p.symbol)
            if px is None:
                continue
            exit_price = float(px) * (1.0 - slip)
            notional = float(p.qty) * exit_price
            fee = notional * float(self._cfg.fee_rate)
            cash += float(notional - fee)
            pnl = (exit_price - float(p.entry_price)) * float(p.qty) - float(p.entry_fee) - float(fee)
            pnl_pct = (exit_price / float(p.entry_price) - 1.0) if p.entry_price > 0 else 0.0

            li = int(last_index_by_symbol.get(p.symbol, p.entry_index))
            duration_bars = int(max(1, li - int(p.entry_index) + 1))
            trades.append(
                Trade(
                    symbol=p.symbol,
                    entry_time=p.entry_time,
                    exit_time=last_t,
                    qty=float(p.qty),
                    entry_price=float(p.entry_price),
                    exit_price=float(exit_price),
                    reason="FORCED_EXIT_EOD",
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    fees_paid=float(p.entry_fee + fee),
                    duration_bars=duration_bars,
                )
            )
            positions.remove(p)

        # Keep the equity curve consistent with forced liquidation.
        if equity_rows:
            equity_rows[-1]["cash"] = float(cash)
            equity_rows[-1]["positions_value"] = 0.0
            equity_rows[-1]["equity"] = float(cash)
            equity_rows[-1]["open_positions"] = 0

        # Write CSV outputs
        equity_curve_path = out_dir / "equity_curve.csv"
        trades_path = out_dir / "trades.csv"

        eq_columns = ["timestamp", "cash", "positions_value", "equity", "open_positions", "drawdown"]
        eq_df = pd.DataFrame(equity_rows, columns=eq_columns)
        eq_df.to_csv(equity_curve_path, index=False)

        tr_columns = [
            "symbol",
            "entry_time",
            "exit_time",
            "qty",
            "entry_price",
            "exit_price",
            "reason",
            "pnl",
            "pnl_pct",
            "fees_paid",
            "duration_bars",
        ]
        tr_df = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "reason": t.reason,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "fees_paid": t.fees_paid,
                    "duration_bars": t.duration_bars,
                }
                for t in trades
            ],
            columns=tr_columns,
        )
        tr_df.to_csv(trades_path, index=False)

        metrics = compute_metrics(
            timestamps=[r["timestamp"] for r in equity_rows],
            equity=[float(r["equity"]) for r in equity_rows],
            trade_pnls=[float(t.pnl) for t in trades],
            trade_entry_ts=[t.entry_time for t in trades],
            trade_exit_ts=[t.exit_time for t in trades],
        )

        return BacktestResult(metrics=metrics, equity_curve_path=equity_curve_path, trades_path=trades_path)

    def _set_cash_local(self, *, locals_dict: dict, name: str, value: float) -> None:
        # Helper for lambda capture; engine uses direct assignments elsewhere.
        locals_dict[name] = value

    def _check_exit_in_bar(
        self,
        *,
        bar: Bar,
        pos: Position,
        fee_rate: float,
        slip: float,
        conservative: bool,
        bar_index: int,
    ) -> tuple[bool, float, Optional[Trade]]:
        # Bot v3.1 semantics:
        # - take_profit is used as tp_initial to activate trailing
        # - exits happen via stop_loss (which may trail above entry)
        hit_sl = float(bar.low) <= float(pos.stop_loss)
        if not hit_sl:
            return False, 0.0, None

        exit_px = float(pos.stop_loss)

        fill_px = float(exit_px) * (1.0 - slip)
        notional = float(pos.qty) * fill_px
        fee_sell = notional * float(fee_rate)

        # PnL is net of sell fee (buy fee already accounted in cash at entry).
        pnl = (fill_px - float(pos.entry_price)) * float(pos.qty) - float(fee_sell)
        pnl_pct = (fill_px / float(pos.entry_price) - 1.0) if pos.entry_price > 0 else 0.0

        duration_bars = int(max(1, int(bar_index) - int(pos.entry_index) + 1))

        reason = "TAKE PROFIT" if float(exit_px) >= float(pos.entry_price) else "STOP LOSS"

        trade = Trade(
            symbol=pos.symbol,
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            qty=float(pos.qty),
            entry_price=float(pos.entry_price),
            exit_price=float(fill_px),
            reason=str(reason),
            pnl=float(pnl - float(pos.entry_fee)),
            pnl_pct=float(pnl_pct),
            fees_paid=float(pos.entry_fee + fee_sell),
            duration_bars=duration_bars,
        )
        # cash delta returned to caller
        cash_delta = float(notional - fee_sell)
        return True, cash_delta, trade
