import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyModel:
    def predict_proba(self, row: pd.DataFrame):
        x = float(row.iloc[0, 0])
        p = max(0.0, min(1.0, x))
        return np.array([[1.0 - p, p]], dtype=float)


class TestEstrategiaV5(unittest.TestCase):
    def _ctx(self, indicators: dict, regime: str = "BULL"):
        from backtesting.bt_types import StrategyContext

        return StrategyContext(
            symbol="BTCUSDT",
            i=10,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            indicators=indicators,
            regime=regime,
            cash=1000.0,
            equity=1000.0,
            positions_open_symbol=0,
            last_entry_time=None,
        )

    def test_prepare_indicators_contains_required_columns(self):
        from estrategia_v5 import BotV5StrategyAdapter

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=260, freq="h"),
                "open": [100 + i * 0.1 for i in range(260)],
                "high": [100.2 + i * 0.1 for i in range(260)],
                "low": [99.8 + i * 0.1 for i in range(260)],
                "close": [100.1 + i * 0.1 for i in range(260)],
                "volume": [1000.0] * 260,
            }
        )

        out = BotV5StrategyAdapter().prepare_indicators(symbol="BTCUSDT", df=df)
        self.assertIn("ema200", out.columns)
        self.assertIn("ema50", out.columns)
        self.assertIn("adx", out.columns)
        self.assertIn("atr", out.columns)
        self.assertIn("regime", out.columns)

    def test_generate_entry_fail_closed_when_model_load_fails(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_fail(_):
            raise RuntimeError("load error")

        strat = BotV5StrategyAdapter(model_path="/tmp/missing.joblib", model_loader=_loader_fail)
        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
        }
        sig = strat.generate_entry(self._ctx(indicators))
        self.assertFalse(sig.should_enter)

    def test_generate_entry_active_vs_shadow(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_active(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob"],
                "operating_threshold": 0.50,
                "signals_enabled": True,
                "operating_mode": "active",
            }

        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
            "feature_prob": 0.9,
        }

        strat_active = BotV5StrategyAdapter(model_path="/tmp/model_active.joblib", model_loader=_loader_active)
        sig_active = strat_active.generate_entry(self._ctx(indicators, regime="BULL"))
        self.assertTrue(sig_active.should_enter)

        def _loader_shadow(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob"],
                "operating_threshold": 0.50,
                "signals_enabled": True,
                "operating_mode": "shadow",
            }

        strat_shadow = BotV5StrategyAdapter(model_path="/tmp/model_shadow.joblib", model_loader=_loader_shadow)
        sig_shadow = strat_shadow.generate_entry(self._ctx(indicators, regime="BULL"))
        self.assertFalse(sig_shadow.should_enter)

    def test_generate_entry_blocks_when_signals_disabled(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_disabled(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob"],
                "operating_threshold": 0.50,
                "signals_enabled": False,
                "operating_mode": "active",
            }

        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
            "feature_prob": 0.9,
        }

        strat = BotV5StrategyAdapter(model_path="/tmp/model_disabled.joblib", model_loader=_loader_disabled)
        sig = strat.generate_entry(self._ctx(indicators, regime="BULL"))
        self.assertFalse(sig.should_enter)

    def test_generate_entry_blocks_in_bear_regime(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_active(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob"],
                "operating_threshold": 0.50,
                "signals_enabled": True,
                "operating_mode": "active",
            }

        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
            "feature_prob": 0.9,
        }

        strat = BotV5StrategyAdapter(model_path="/tmp/model_bear.joblib", model_loader=_loader_active)
        sig = strat.generate_entry(self._ctx(indicators, regime="BEAR"))
        self.assertFalse(sig.should_enter)

    def test_generate_entry_blocks_when_features_missing(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_active(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob", "feature_extra"],
                "operating_threshold": 0.50,
                "signals_enabled": True,
                "operating_mode": "active",
            }

        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
            "feature_prob": 0.9,
        }

        strat = BotV5StrategyAdapter(model_path="/tmp/model_missing_features.joblib", model_loader=_loader_active)
        sig = strat.generate_entry(self._ctx(indicators, regime="BULL"))
        self.assertFalse(sig.should_enter)

    def test_generate_entry_respects_threshold_override(self):
        from estrategia_v5 import BotV5StrategyAdapter

        def _loader_active(_):
            return {
                "model": _DummyModel(),
                "features": ["feature_prob"],
                "operating_threshold": 0.50,
                "signals_enabled": True,
                "operating_mode": "active",
            }

        indicators = {
            "close": 110.0,
            "ema200": 100.0,
            "ema50": 105.0,
            "adx": 30.0,
            "feature_prob": 0.9,
        }

        strat = BotV5StrategyAdapter(
            model_path="/tmp/model_override.joblib",
            model_loader=_loader_active,
            threshold_override=0.95,
        )
        sig = strat.generate_entry(self._ctx(indicators, regime="BULL"))
        self.assertFalse(sig.should_enter)

    def test_compute_risk_levels_atr_dynamic_values(self):
        from estrategia_v5 import BotV5StrategyAdapter

        strat = BotV5StrategyAdapter()
        tp, sl, extra = strat.compute_risk_levels(
            symbol="BTCUSDT",
            regime="BULL",
            buy_price=100.0,
            indicators_row={"atr": 2.0},
        )

        self.assertAlmostEqual(tp, 105.0, places=8)
        self.assertAlmostEqual(sl, 97.0, places=8)
        self.assertEqual(extra["tp_initial"], tp)
        self.assertEqual(extra["trailing_sl_atr_mult"], 1.2)

    def test_update_trailing_moves_stop_when_active(self):
        from backtesting.bt_types import Position, Bar
        from estrategia_v5 import BotV5StrategyAdapter

        strat = BotV5StrategyAdapter()
        pos = Position(
            symbol="BTCUSDT",
            qty=1.0,
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            entry_index=1,
            entry_price=100.0,
            regime="BULL",
            take_profit=105.0,
            stop_loss=97.0,
            tp_initial=105.0,
            trailing_active=False,
            max_price=100.0,
            trailing_sl_atr_mult=1.2,
        )

        bar = Bar(
            timestamp=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
            open=105.0,
            high=106.0,
            low=104.0,
            close=106.0,
            volume=10.0,
        )

        strat.update_trailing(symbol="BTCUSDT", position=pos, bar=bar, indicators_row={"atr": 2.0})

        self.assertTrue(pos.trailing_active)
        self.assertGreater(pos.max_price, 100.0)
        self.assertGreater(pos.stop_loss, 97.0)


if __name__ == "__main__":
    unittest.main()
