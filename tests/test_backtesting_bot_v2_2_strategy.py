import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestBacktestingBotV2_2Strategy(unittest.TestCase):
    def _df(self, n: int = 260, start: datetime | None = None) -> pd.DataFrame:
        start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        price = 100.0
        for i in range(n):
            ts = start + timedelta(hours=i)
            # mild oscillation
            o = price
            c = price + (0.2 if i % 2 == 0 else -0.1)
            h = max(o, c) + 0.2
            l = min(o, c) - 0.2
            rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000.0})
            price = c
        return pd.DataFrame(rows)

    def test_adapter_prepares_regime_and_change(self):
        from backtesting.strategies import BotV2_2StrategyAdapter

        df = self._df()
        strat = BotV2_2StrategyAdapter()
        out = strat.prepare_indicators(symbol="BTCUSDT", df=df)
        self.assertIn("regime", out.columns)
        self.assertIn("change_5", out.columns)
        self.assertEqual(len(out), len(df))


if __name__ == "__main__":
    unittest.main()
