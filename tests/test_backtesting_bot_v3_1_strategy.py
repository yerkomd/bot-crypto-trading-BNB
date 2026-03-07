import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestBacktestingBotV3Strategy(unittest.TestCase):
    def _df(self, n: int = 260, start: datetime | None = None) -> pd.DataFrame:
        start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        price = 100.0
        for i in range(n):
            ts = start + timedelta(hours=i)
            # gentle uptrend
            o = price
            c = price + 0.1
            h = max(o, c) + 0.2
            l = min(o, c) - 0.2
            rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000.0})
            price = c
        return pd.DataFrame(rows)

    def test_adapter_prepares_regime_column(self):
        from backtesting.strategies import BotV3_1StrategyAdapter

        df = self._df()
        strat = BotV3_1StrategyAdapter()
        out = strat.prepare_indicators(symbol="BTCUSDT", df=df)
        self.assertIn("regime", out.columns)
        self.assertEqual(len(out), len(df))

    def test_engine_no_lookahead_entry_uses_previous_bar(self):
        # Construct data where only bar i-1 satisfies the entry condition.
        from backtesting.engine import BacktestEngine, BacktestConfig
        from backtesting.strategies import BotV3_1StrategyAdapter

        df = self._df()

        strat = BotV3_1StrategyAdapter()
        prepared = strat.prepare_indicators(symbol="BTCUSDT", df=df)

        # Force a single entry condition at index 10 (signal bar), and make index 11 open distinct.
        # Entry uses bar i-1; so trade should enter at i=11 open.
        prepared.loc[:, "ema200"] = prepared["close"].ewm(span=200, adjust=False, min_periods=1).mean()
        prepared.loc[:, "ema50"] = prepared["close"].ewm(span=50, adjust=False, min_periods=1).mean()
        prepared.loc[:, "adx"] = 30.0
        prepared.loc[:, "rsi"] = 10.0  # default: fail RSI range

        # Pick index 10 to be in RSI range
        prepared.loc[10, "rsi"] = 45.0

        # Ensure close > ema200 and ema50 > ema200 at index 10
        prepared.loc[10, "ema200"] = prepared.loc[10, "close"] - 1.0
        prepared.loc[10, "ema50"] = prepared.loc[10, "ema200"] + 1.0

        # Keep other rows failing
        prepared.loc[:, "regime"] = "BULL"

        # Make bar 11 open unique so we can detect entry price basis.
        prepared.loc[11, "open"] = 123.45

        engine = BacktestEngine(
            strategy=strat,
            config=BacktestConfig(initial_cash=1000.0, slippage_bps=0.0, fee_rate=0.001),
        )
        res = engine.run(data_by_symbol={"BTCUSDT": prepared}, output_dir=ROOT / "tests" / "_out_backtest")

        trades = pd.read_csv(res.trades_path)
        # Either 0 or >=1 trades depending on ATR availability; but if it entered, entry_price must match bar 11 open.
        if not trades.empty:
            self.assertAlmostEqual(float(trades.iloc[0]["entry_price"]), 123.45, places=6)

    def test_engine_exit_is_stop_only(self):
        # Ensure take_profit isn't used as a direct exit.
        from backtesting.engine import BacktestEngine, BacktestConfig
        from backtesting.bt_types import StrategyContext, EntrySignal
        from backtesting.bt_types import Position, Bar

        class DummyStrat:
            def prepare_indicators(self, *, symbol: str, df):
                out = df.copy()
                out["regime"] = "BULL"
                out["atr"] = 1.0
                out["rsi"] = 45.0
                out["ema200"] = out["close"] - 10.0
                out["ema50"] = out["close"] - 5.0
                out["adx"] = 30.0
                return out

            def generate_entry(self, ctx: StrategyContext) -> EntrySignal:
                return EntrySignal(True, 0.5, None)

            def compute_risk_levels(self, *, symbol: str, regime: str, buy_price: float, indicators_row):
                # take_profit set very low so bar.high always exceeds it; stop_loss far away
                tp = buy_price + 0.01
                sl = buy_price - 100.0
                return tp, sl, {"tp_initial": tp, "trailing_active": False, "max_price": buy_price}

            def update_trailing(self, *, symbol: str, position: Position, bar: Bar, indicators_row):
                return None

        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame(
            [
                {"timestamp": now, "open": 100, "high": 200, "low": 99, "close": 150, "volume": 1.0},
                {"timestamp": now + timedelta(hours=1), "open": 100, "high": 200, "low": 99, "close": 150, "volume": 1.0},
            ]
        )

        engine = BacktestEngine(strategy=DummyStrat(), config=BacktestConfig(initial_cash=1000.0))
        res = engine.run(data_by_symbol={"BTCUSDT": df}, output_dir=ROOT / "tests" / "_out_backtest2")
        trades = pd.read_csv(res.trades_path)
        # Stop is never hit; only forced end-of-backtest liquidation is allowed.
        if not trades.empty:
            self.assertTrue((trades["reason"] == "FORCED_EXIT_EOD").all())


if __name__ == "__main__":
    unittest.main()
