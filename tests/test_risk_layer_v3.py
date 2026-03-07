import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))


import risk_layer_v3 as rl3


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((str(sql), params))

    def fetchall(self):
        return list(self.rows)


class FakeDB:
    def __init__(self, rows=None):
        self.cursor = FakeCursor(rows=rows)
        self.run_calls = 0

    def run(self, fn, **_kwargs):
        self.run_calls += 1
        return fn(self.cursor)


class DummyStopEvent:
    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True


class DummyEventLogger:
    def __init__(self):
        self.events = []

    def insert(self, **kwargs):
        self.events.append(kwargs)


class RiskLayerV3UnitTests(unittest.TestCase):
    def test_volatility_position_sizer_computes_capped_qty(self):
        with patch.dict(
            os.environ,
            {
                "RISK_V3_POSITION_SIZER_ENABLED": "true",
                "RISK_PER_TRADE_FRAC": "0.01",
                "MAX_POSITION_CAP_FRAC": "0.10",
                "MIN_POSITION_USDT": "10",
                "POSITION_SIZER_ATR_MULTIPLIER": "1.0",
            },
            clear=False,
        ):
            sizer = rl3.VolatilityPositionSizer()
            qty = sizer.compute_position_size("BTCUSDT", equity_total=1000.0, atr=0.1, price=100.0)
            # raw usdt = (1000*0.01)/(0.1*1)=100 -> cap(10%)=100 -> qty=1
            self.assertAlmostEqual(qty, 1.0, places=8)

    def test_volatility_position_sizer_disabled_returns_zero(self):
        with patch.dict(os.environ, {"RISK_V3_POSITION_SIZER_ENABLED": "false"}, clear=False):
            sizer = rl3.VolatilityPositionSizer()
            qty = sizer.compute_position_size("BTCUSDT", equity_total=1000.0, atr=1.0, price=100.0)
            self.assertEqual(qty, 0.0)

    def test_portfolio_correlation_risk_blocks_when_combined_exposure_exceeds_limit(self):
        with patch.dict(
            os.environ,
            {
                "RISK_V3_CORRELATION_ENABLED": "true",
                "CORRELATION_WINDOW_DAYS": "30",
                "CORRELATION_THRESHOLD": "0.8",
                "CORRELATION_MAX_COMBINED_EXPOSURE_FRAC": "0.25",
                "CORRELATION_CACHE_SECONDS": "0",
            },
            clear=False,
        ):
            corr_risk = rl3.PortfolioCorrelationRisk(db=object(), symbols=["BTCUSDT", "ETHUSDT"])

            close = list(range(100, 141))  # 41 points -> 40 log returns

            def fake_read_klines_df(*, db, symbol, interval, limit):
                _ = (db, symbol, interval, limit)
                return pd.DataFrame({"close": close})

            with patch("risk_layer_v3.read_klines_df", side_effect=fake_read_klines_df):
                positions = [
                    {"symbol": "BTCUSDT", "amount": 2.0, "current_price": 100.0},
                    {"symbol": "ETHUSDT", "amount": 1.5, "current_price": 100.0},
                ]
                allowed, reason = corr_risk.can_open("BTCUSDT", current_positions=positions, equity_total=1000.0)

            self.assertFalse(allowed)
            self.assertIsNotNone(reason)
            self.assertIn("correlation_exposure", str(reason))

    def test_intraday_var_monitor_blocks_when_var_exceeds_max(self):
        rows = [(None, v) for v in [100, 95, 90, 80, 72, 65, 60, 58, 54, 49, 45, 41]]
        db = FakeDB(rows=rows)

        with patch.dict(
            os.environ,
            {
                "RISK_V3_VAR_ENABLED": "true",
                "VAR_WINDOW_DAYS": "10",
                "VAR_CONFIDENCE": "0.95",
                "MAX_VAR_FRAC": "0.05",
                "VAR_CACHE_SECONDS": "0",
            },
            clear=False,
        ):
            mon = rl3.IntradayVaRMonitor(db=db)
            allowed, reason = mon.can_open(equity_total=1000.0)

        self.assertFalse(allowed)
        self.assertIsNotNone(reason)
        self.assertIn("intraday_var", str(reason))
        self.assertGreaterEqual(db.run_calls, 1)

    def test_slippage_monitor_sets_stop_event_after_consecutive_breaches(self):
        stop_event = DummyStopEvent()
        event_logger = DummyEventLogger()
        telegram_msgs = []

        with patch.dict(
            os.environ,
            {
                "RISK_V3_SLIPPAGE_ENABLED": "true",
                "SLIPPAGE_MAX_FRAC": "0.005",
                "SLIPPAGE_MAX_CONSECUTIVE": "3",
            },
            clear=False,
        ):
            mon = rl3.SlippageMonitor(
                stop_event=stop_event,
                event_logger=event_logger,
                send_telegram=lambda m: telegram_msgs.append(m),
            )

            mon.record_fill(expected_price=100.0, fill_price=101.0)  # 1.0%
            self.assertFalse(stop_event.is_set())
            mon.record_fill(expected_price=100.0, fill_price=100.7)  # 0.7%
            self.assertFalse(stop_event.is_set())
            mon.record_fill(expected_price=100.0, fill_price=100.8)  # 0.8%

        self.assertTrue(stop_event.is_set())
        self.assertGreaterEqual(len(event_logger.events), 3)
        self.assertTrue(any("RISK_V3_BLOCK" in str(m) for m in telegram_msgs))

    def test_equity_regime_filter_reduce_and_block_modes(self):
        db = FakeDB(rows=[])

        with patch.dict(
            os.environ,
            {
                "RISK_V3_EQUITY_REGIME_ENABLED": "true",
                "EQUITY_REGIME_MODE": "reduce",
                "EQUITY_REGIME_REDUCTION_FRAC": "0.5",
            },
            clear=False,
        ):
            flt_reduce = rl3.EquityRegimeFilter(db=db)
            with patch.object(flt_reduce, "_is_bearish", return_value=True):
                self.assertAlmostEqual(flt_reduce.adjust_position_size(200.0), 100.0)
                can_open, reason = flt_reduce.can_open()
                self.assertTrue(can_open)
                self.assertIsNone(reason)

        with patch.dict(
            os.environ,
            {
                "RISK_V3_EQUITY_REGIME_ENABLED": "true",
                "EQUITY_REGIME_MODE": "block",
            },
            clear=False,
        ):
            flt_block = rl3.EquityRegimeFilter(db=db)
            with patch.object(flt_block, "_is_bearish", return_value=True):
                can_open, reason = flt_block.can_open()
                self.assertFalse(can_open)
                self.assertIn("equity_regime_bearish", str(reason))


if __name__ == "__main__":
    unittest.main()
