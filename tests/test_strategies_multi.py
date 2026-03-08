"""
tests/test_strategies_multi.py — Unit tests for strategies_multi.py
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import ta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies_multi import (
    FundingArbitrageStrategy,
    MeanReversionStrategy,
    MultiStrategyEngine,
    StrategySignal,
    TrendFollowingStrategy,
    VolatilityBreakoutStrategy,
    build_default_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 260, trend: bool = True) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with basic indicators pre-computed."""
    np.random.seed(0)
    base = 50_000.0
    if trend:
        price = base + np.cumsum(np.abs(np.random.randn(n)) * 100)
    else:
        price = base + np.random.randn(n) * 200
    df = pd.DataFrame({
        "open":   price * 0.999,
        "high":   price * 1.003,
        "low":    price * 0.997,
        "close":  price,
        "volume": np.random.uniform(100, 500, n),
    })
    df["ema50"]  = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["adx"]    = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df["rsi"]    = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi14"]  = df["rsi"]
    return df


# ---------------------------------------------------------------------------
# TrendFollowingStrategy
# ---------------------------------------------------------------------------

class TestTrendFollowingStrategy(unittest.TestCase):
    def setUp(self):
        self.strat = TrendFollowingStrategy()

    def test_eligible_symbols_uppercase(self):
        self.assertTrue(all(s == s.upper() for s in self.strat.ELIGIBLE_SYMBOLS))

    def test_bear_regime_returns_no_entry(self):
        df = _make_df()
        sig = self.strat.signal(df, "BTCUSDT", "BEAR", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_not_eligible_symbol_returns_false_via_engine(self):
        """is_eligible must reject symbols not in ELIGIBLE_SYMBOLS."""
        self.assertFalse(self.strat.is_eligible("SOLUSDT"))

    def test_eligible_btcusdt(self):
        self.assertTrue(self.strat.is_eligible("BTCUSDT"))

    def test_signal_nan_indicators_returns_no_entry(self):
        df = pd.DataFrame({
            "open": [100.0] * 10, "high": [101.0] * 10,
            "low": [99.0] * 10, "close": [100.0] * 10, "volume": [1.0] * 10,
        })
        # No ema columns → NaN
        sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_signal_returns_strategy_name(self):
        df = _make_df()
        sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertEqual(sig.strategy_name, "trend_following")


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self):
        self.strat = MeanReversionStrategy()

    def test_eligible_symbols_uppercase(self):
        """Regression: ELIGIBLE_SYMBOLS must be uppercase after fix."""
        for sym in self.strat.ELIGIBLE_SYMBOLS:
            if not sym.startswith("RE:"):
                self.assertEqual(sym, sym.upper(), f"Not uppercase: {sym}")

    def test_bear_regime_returns_no_entry(self):
        df = _make_df()
        sig = self.strat.signal(df, "SOLUSDT", "BEAR", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_btcusdt_not_eligible_by_default(self):
        self.assertFalse(self.strat.is_eligible("BTCUSDT"))

    def test_solusdt_eligible_by_default(self):
        self.assertTrue(self.strat.is_eligible("SOLUSDT"))

    def test_rsi_crossing_up_requires_previous_below_oversold(self):
        """
        Regression: rsi_crossing_up must require rsi_prev <= rsi_os, not just any uptick.
        Construct a df where rsi is 50 (well above oversold) and ticking up —
        this should NOT fire a signal.
        """
        df = _make_df(trend=False)
        # Patch RSI to values that are above oversold (50 → 51)
        # by making the df produce RSI > rsi_os at iloc[-3] and iloc[-2]
        # We verify by checking the signal is False when rsi_prev > rsi_os
        with patch.object(
            ta.momentum.RSIIndicator, "rsi",
            return_value=pd.Series([50.0] * 257 + [50.5, 51.0, 51.5])
        ):
            sig = self.strat.signal(df, "SOLUSDT", "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_short_df_returns_no_entry_gracefully(self):
        # Raw OHLCV with only 5 rows — strategy must not raise
        df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5,
            "low": [99.0] * 5, "close": [100.0] * 5, "volume": [1.0] * 5,
        })
        sig = self.strat.signal(df, "SOLUSDT", "BULL", 1000.0)
        self.assertFalse(sig.should_enter)


# ---------------------------------------------------------------------------
# FundingArbitrageStrategy
# ---------------------------------------------------------------------------

class TestFundingArbitrageStrategy(unittest.TestCase):
    def setUp(self):
        self.strat = FundingArbitrageStrategy()
        # Clear shared cache between tests
        FundingArbitrageStrategy._CACHE.clear()

    def tearDown(self):
        self.strat.close()

    def test_bear_regime_blocks_entry(self):
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=0.0):
            sig = self.strat.signal(df, "BTCUSDT", "BEAR", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_api_failure_is_fail_open(self):
        """Regression: API failure must NOT block spot trading (fail-open)."""
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=None):
            sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_excessive_positive_funding_blocks_entry(self):
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=0.002):
            sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_neutral_funding_passes_entry(self):
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=0.0001):
            sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_negative_funding_passes_entry(self):
        """Regression: negative funding should pass (not force entry) for spot."""
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=-0.001):
            sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_position_size_frac_is_none(self):
        """Funding filter must not suggest a position size."""
        df = _make_df()
        with patch.object(self.strat, "_fetch_funding_rate", return_value=-0.001):
            sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertIsNone(sig.position_size_frac)

    def test_cache_is_thread_safe(self):
        """_CACHE_LOCK must exist and be a threading.Lock."""
        import threading
        self.assertIsInstance(FundingArbitrageStrategy._CACHE_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# VolatilityBreakoutStrategy
# ---------------------------------------------------------------------------

class TestVolatilityBreakoutStrategy(unittest.TestCase):
    def setUp(self):
        self.strat = VolatilityBreakoutStrategy()

    def test_bear_regime_blocks_entry(self):
        df = _make_df()
        sig = self.strat.signal(df, "BTCUSDT", "BEAR", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_bb_expansion_lag_is_configurable(self):
        """Regression: bb_expansion_lag must be an attribute (not hardcoded)."""
        self.assertTrue(hasattr(self.strat, "bb_expansion_lag"))
        self.assertIsInstance(self.strat.bb_expansion_lag, int)

    def test_short_df_returns_gracefully(self):
        # Raw OHLCV with only 10 rows — strategy must not raise
        df = pd.DataFrame({
            "open": [100.0] * 10, "high": [101.0] * 10,
            "low": [99.0] * 10, "close": [100.0] * 10, "volume": [1.0] * 10,
        })
        sig = self.strat.signal(df, "BTCUSDT", "BULL", 1000.0)
        self.assertFalse(sig.should_enter)


# ---------------------------------------------------------------------------
# MultiStrategyEngine — consensus modes
# ---------------------------------------------------------------------------

class TestMultiStrategyEngine(unittest.TestCase):
    def _make_signal(self, enter: bool, size: float = 0.05) -> StrategySignal:
        return StrategySignal(
            should_enter=enter,
            strategy_name="mock",
            position_size_frac=size if enter else None,
        )

    def _engine_with_mocks(self, results: list[bool], mode: str) -> MultiStrategyEngine:
        strats = []
        for i, enters in enumerate(results):
            s = MagicMock(spec=TrendFollowingStrategy)
            s.name = f"mock_{i}"
            s.is_eligible.return_value = True
            s.signal.return_value = self._make_signal(enters, 0.04 + i * 0.01)
            strats.append(s)
        return MultiStrategyEngine(strategies=strats, mode=mode)

    def test_mode_any_one_positive_enters(self):
        engine = self._engine_with_mocks([False, False, True], "ANY")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_mode_any_all_negative_blocks(self):
        engine = self._engine_with_mocks([False, False, False], "ANY")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_mode_majority_passes(self):
        engine = self._engine_with_mocks([True, True, False, False], "MAJORITY")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        # 2 out of 4 is not > 2 (need strictly more than half)
        self.assertFalse(sig.should_enter)

    def test_mode_majority_three_of_four_passes(self):
        engine = self._engine_with_mocks([True, True, True, False], "MAJORITY")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_mode_all_requires_all(self):
        engine = self._engine_with_mocks([True, True, False], "ALL")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_mode_all_all_positive_passes(self):
        engine = self._engine_with_mocks([True, True, True], "ALL")
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertTrue(sig.should_enter)

    def test_position_size_is_minimum_not_average(self):
        """Regression: position_size_frac must be min() of positive strategies."""
        engine = self._engine_with_mocks([True, True], "ANY")
        # mock_0 → size 0.04, mock_1 → size 0.05
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertTrue(sig.should_enter)
        self.assertAlmostEqual(sig.position_size_frac, 0.04, places=8)

    def test_invalid_mode_falls_back_to_any(self):
        engine = MultiStrategyEngine(strategies=[], mode="INVALID")
        self.assertEqual(engine.mode, "ANY")

    def test_no_eligible_strategies_returns_false(self):
        s = MagicMock()
        s.name = "mock_ineligible"   # must be str; MagicMock.name is special
        s.is_eligible.return_value = False
        engine = MultiStrategyEngine(strategies=[s], mode="ANY")
        sig = engine.evaluate("SOLUSDT", MagicMock(), "BULL", 1000.0)
        self.assertFalse(sig.should_enter)

    def test_strategy_exception_is_caught(self):
        s = MagicMock(spec=TrendFollowingStrategy)
        s.name = "broken"
        s.is_eligible.return_value = True
        s.signal.side_effect = RuntimeError("oops")
        engine = MultiStrategyEngine(strategies=[s], mode="ANY")
        # Must not raise
        sig = engine.evaluate("BTCUSDT", MagicMock(), "BULL", 1000.0)
        self.assertFalse(sig.should_enter)


# ---------------------------------------------------------------------------
# build_default_engine
# ---------------------------------------------------------------------------

class TestBuildDefaultEngine(unittest.TestCase):
    def test_returns_four_strategies(self):
        engine = build_default_engine()
        self.assertEqual(len(engine.strategies), 4)

    def test_default_mode_is_majority(self):
        """Regression: default mode must be MAJORITY, not ANY."""
        import os
        old = os.environ.pop("STRATEGY_MODE", None)
        try:
            engine = build_default_engine()
            self.assertEqual(engine.mode, "MAJORITY")
        finally:
            if old is not None:
                os.environ["STRATEGY_MODE"] = old

    def test_strategy_names(self):
        engine = build_default_engine()
        names = [s.name for s in engine.strategies]
        self.assertIn("trend_following", names)
        self.assertIn("mean_reversion", names)
        self.assertIn("funding_arb", names)
        self.assertIn("vol_breakout", names)


if __name__ == "__main__":
    unittest.main()
