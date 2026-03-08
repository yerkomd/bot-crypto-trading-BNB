"""
tests/test_strategy_engine_and_portfolio.py
Unit tests for strategy_engine.py and portfolio_manager.py.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from strategies.base_strategy import BaseStrategy, MarketState, Signal
from strategy_engine import StrategyEngine
from portfolio_manager import PortfolioManager, PortfolioOrder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(symbol: str = "BTCUSDT", regime: str = "BULL") -> MarketState:
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.0], "volume": [1.0],
    })
    return MarketState(
        symbol=symbol, df=df, regime=regime,
        balance=1000.0, equity=1000.0, open_positions=[],
        indicators={"close": 100.0, "ema200": 90.0, "ema50": 95.0, "adx": 30.0},
    )


def _buy_strat(strategy_id: str = "s1", size: float = 0.05, conf: float = 0.8,
               eligible: bool = True, weight: float = 1.0) -> BaseStrategy:
    s = MagicMock(spec=BaseStrategy)
    s.strategy_id = strategy_id
    s.weight = weight
    s.is_eligible.return_value = eligible
    s.generate_signal.return_value = Signal(
        symbol="BTCUSDT", side="BUY", size_frac=size,
        strategy_id=strategy_id, confidence=conf,
    )
    return s


def _hold_strat(strategy_id: str = "s_hold", eligible: bool = True,
                weight: float = 1.0) -> BaseStrategy:
    s = MagicMock(spec=BaseStrategy)
    s.strategy_id = strategy_id
    s.weight = weight
    s.is_eligible.return_value = eligible
    s.generate_signal.return_value = Signal.hold("BTCUSDT", strategy_id)
    return s


def _sell_strat(strategy_id: str = "s_sell", conf: float = 0.8,
                eligible: bool = True, weight: float = 1.0) -> BaseStrategy:
    s = MagicMock(spec=BaseStrategy)
    s.strategy_id = strategy_id
    s.weight = weight
    s.is_eligible.return_value = eligible
    s.generate_signal.return_value = Signal(
        symbol="BTCUSDT", side="SELL", size_frac=0.0,
        strategy_id=strategy_id, confidence=conf,
    )
    return s


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------

class TestMarketState(unittest.TestCase):
    def test_fields_accessible(self):
        state = _make_state()
        self.assertEqual(state.symbol, "BTCUSDT")
        self.assertEqual(state.regime, "BULL")
        self.assertEqual(state.balance, 1000.0)
        self.assertIsInstance(state.indicators, dict)

    def test_open_positions_defaults_empty(self):
        state = _make_state()
        self.assertEqual(state.open_positions, [])


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal(unittest.TestCase):
    def test_is_buy(self):
        sig = Signal("BTCUSDT", "BUY", 0.05, "s1")
        self.assertTrue(sig.is_buy)
        self.assertFalse(sig.is_sell)
        self.assertFalse(sig.is_hold)

    def test_is_hold(self):
        sig = Signal.hold("BTCUSDT", "s1", reason="bear")
        self.assertTrue(sig.is_hold)
        self.assertFalse(sig.is_buy)
        self.assertEqual(sig.size_frac, 0.0)
        self.assertEqual(sig.meta["reason"], "bear")

    def test_is_sell(self):
        sig = Signal("BTCUSDT", "SELL", 0.0, "s1")
        self.assertTrue(sig.is_sell)


# ---------------------------------------------------------------------------
# StrategyEngine
# ---------------------------------------------------------------------------

class TestStrategyEngine(unittest.TestCase):
    def test_collect_returns_signal_from_eligible(self):
        strat = _buy_strat()
        engine = StrategyEngine([strat])
        signals = engine.collect(_make_state())
        self.assertEqual(len(signals), 1)
        self.assertTrue(signals[0].is_buy)

    def test_ineligible_strategy_is_skipped(self):
        strat = _buy_strat(eligible=False)
        engine = StrategyEngine([strat])
        signals = engine.collect(_make_state())
        self.assertEqual(signals, [])

    def test_exception_in_strategy_is_caught(self):
        strat = MagicMock(spec=BaseStrategy)
        strat.strategy_id = "broken"
        strat.is_eligible.return_value = True
        strat.generate_signal.side_effect = RuntimeError("boom")
        engine = StrategyEngine([strat])
        # Must not raise
        signals = engine.collect(_make_state())
        self.assertEqual(signals, [])

    def test_multiple_strategies_all_collected(self):
        s1 = _buy_strat("s1")
        s2 = _hold_strat("s2")
        engine = StrategyEngine([s1, s2])
        signals = engine.collect(_make_state())
        self.assertEqual(len(signals), 2)

    def test_close_calls_each_strategy_close(self):
        s1 = _buy_strat("s1")
        s2 = _buy_strat("s2")
        engine = StrategyEngine([s1, s2])
        engine.close()
        s1.close.assert_called_once()
        s2.close.assert_called_once()

    def test_empty_engine_returns_empty_list(self):
        engine = StrategyEngine([])
        signals = engine.collect(_make_state())
        self.assertEqual(signals, [])


# ---------------------------------------------------------------------------
# PortfolioManager — BUY scenarios
# ---------------------------------------------------------------------------

class TestPortfolioManagerBuy(unittest.TestCase):
    def _pm(self, strategies=None, buy_thr=1.0, veto=True):
        strategies = strategies or []
        return PortfolioManager(strategies=strategies, buy_threshold=buy_thr,
                                sell_threshold=1.0, veto_on_hold=veto)

    def test_single_buy_signal_above_threshold(self):
        pm = self._pm(buy_thr=0.5, veto=False)
        signals = [Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=0.8)]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "BUY")
        self.assertAlmostEqual(order.size_frac, 0.05, places=6)

    def test_buy_size_is_minimum_of_active_signals(self):
        pm = PortfolioManager(
            strategies=[_buy_strat("s1", weight=1.0), _buy_strat("s2", weight=1.0)],
            buy_threshold=0.5, sell_threshold=1.0, veto_on_hold=False,
        )
        signals = [
            Signal("BTCUSDT", "BUY", 0.08, "s1", confidence=1.0),
            Signal("BTCUSDT", "BUY", 0.04, "s2", confidence=1.0),
        ]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "BUY")
        self.assertAlmostEqual(order.size_frac, 0.04, places=6)

    def test_score_weighted_by_confidence(self):
        """Estrategias con mayor weight*confidence deben dominar el score."""
        strats = [_buy_strat("s1", conf=1.0, weight=2.0)]
        pm = PortfolioManager(strategies=strats, buy_threshold=1.5, veto_on_hold=False)
        signals = [Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=1.0)]
        order = pm.decide("BTCUSDT", signals)
        # score = 2.0 * 1.0 = 2.0 >= 1.5 → BUY
        self.assertEqual(order.side, "BUY")

    def test_no_signals_returns_hold(self):
        pm = self._pm()
        order = pm.decide("BTCUSDT", [])
        self.assertEqual(order.side, "HOLD")
        self.assertEqual(order.meta.get("reason"), "no_signals")

    def test_triggered_by_populated(self):
        pm = self._pm(buy_thr=0.5, veto=False)
        signals = [Signal("BTCUSDT", "BUY", 0.05, "trend", confidence=0.9)]
        order = pm.decide("BTCUSDT", signals)
        self.assertIn("trend", order.triggered_by)


# ---------------------------------------------------------------------------
# PortfolioManager — HOLD veto
# ---------------------------------------------------------------------------

class TestPortfolioManagerVeto(unittest.TestCase):
    def test_hold_signal_vetoes_buy(self):
        """Con veto_on_hold=True, un HOLD bloquea la entrada aunque haya BUY."""
        pm = PortfolioManager(
            strategies=[_buy_strat("s1"), _hold_strat("s2")],
            buy_threshold=0.1, sell_threshold=1.0, veto_on_hold=True,
        )
        signals = [
            Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=1.0),
            Signal.hold("BTCUSDT", "s2", reason="funding_blocked"),
        ]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "HOLD")
        self.assertIn("s2", order.vetoed_by)

    def test_hold_signal_ignored_when_veto_disabled(self):
        pm = PortfolioManager(
            strategies=[_buy_strat("s1"), _hold_strat("s2")],
            buy_threshold=0.1, sell_threshold=1.0, veto_on_hold=False,
        )
        signals = [
            Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=1.0),
            Signal.hold("BTCUSDT", "s2"),
        ]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "BUY")

    def test_all_hold_returns_hold(self):
        pm = PortfolioManager(strategies=[], veto_on_hold=True)
        signals = [Signal.hold("BTCUSDT", "s1"), Signal.hold("BTCUSDT", "s2")]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "HOLD")


# ---------------------------------------------------------------------------
# PortfolioManager — SELL
# ---------------------------------------------------------------------------

class TestPortfolioManagerSell(unittest.TestCase):
    def test_sell_above_threshold(self):
        pm = PortfolioManager(
            strategies=[_sell_strat("s1")],
            buy_threshold=1.0, sell_threshold=0.5, veto_on_hold=False,
        )
        signals = [Signal("BTCUSDT", "SELL", 0.0, "s1", confidence=0.9)]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "SELL")

    def test_buy_wins_over_sell_when_higher_score(self):
        pm = PortfolioManager(
            strategies=[
                _buy_strat("s1", conf=0.9, weight=2.0),
                _sell_strat("s2", conf=0.5, weight=1.0),
            ],
            buy_threshold=1.0, sell_threshold=0.5, veto_on_hold=False,
        )
        signals = [
            Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=0.9),
            Signal("BTCUSDT", "SELL", 0.0, "s2", confidence=0.5),
        ]
        order = pm.decide("BTCUSDT", signals)
        # score_buy = 2.0*0.9=1.8, score_sell=1.0*0.5=0.5 → BUY
        self.assertEqual(order.side, "BUY")


# ---------------------------------------------------------------------------
# PortfolioManager — below threshold
# ---------------------------------------------------------------------------

class TestPortfolioManagerBelowThreshold(unittest.TestCase):
    def test_buy_below_threshold_returns_hold(self):
        pm = PortfolioManager(
            strategies=[], buy_threshold=5.0, sell_threshold=5.0, veto_on_hold=False,
        )
        signals = [Signal("BTCUSDT", "BUY", 0.05, "s1", confidence=0.3)]
        order = pm.decide("BTCUSDT", signals)
        self.assertEqual(order.side, "HOLD")
        self.assertEqual(order.meta.get("reason"), "below_threshold")

    def test_should_enter_property_true_for_buy(self):
        order = PortfolioOrder(symbol="BTCUSDT", side="BUY", size_frac=0.05, score=1.5)
        self.assertTrue(order.should_enter)

    def test_should_enter_property_false_for_hold(self):
        order = PortfolioOrder(symbol="BTCUSDT", side="HOLD", size_frac=0.0, score=0.0)
        self.assertFalse(order.should_enter)

    def test_should_exit_property(self):
        order = PortfolioOrder(symbol="BTCUSDT", side="SELL", size_frac=0.0, score=0.5)
        self.assertTrue(order.should_exit)


# ---------------------------------------------------------------------------
# StrategyEngine integration with PortfolioManager
# ---------------------------------------------------------------------------

class TestEnginePortfolioIntegration(unittest.TestCase):
    def test_two_buy_one_hold_with_veto(self):
        """Dos BUY + un HOLD con veto → debe bloquear."""
        s1 = _buy_strat("s1", conf=0.9, weight=2.0)
        s2 = _buy_strat("s2", conf=0.8, weight=1.5)
        s3 = _hold_strat("funding", weight=0.5)

        engine = StrategyEngine([s1, s2, s3])
        pm = PortfolioManager(
            strategies=[s1, s2, s3],
            buy_threshold=1.0, veto_on_hold=True,
        )

        state = _make_state()
        signals = engine.collect(state)
        order = pm.decide(state.symbol, signals)
        self.assertEqual(order.side, "HOLD")
        self.assertIn("funding", order.vetoed_by)

    def test_two_buy_one_hold_without_veto(self):
        """Dos BUY + un HOLD sin veto → debe entrar."""
        s1 = _buy_strat("s1", conf=0.9, weight=2.0)
        s2 = _buy_strat("s2", conf=0.8, weight=1.5)
        s3 = _hold_strat("funding", weight=0.5)

        engine = StrategyEngine([s1, s2, s3])
        pm = PortfolioManager(
            strategies=[s1, s2, s3],
            buy_threshold=1.0, veto_on_hold=False,
        )

        state = _make_state()
        signals = engine.collect(state)
        order = pm.decide(state.symbol, signals)
        self.assertEqual(order.side, "BUY")
        self.assertIn("s1", order.triggered_by)
        self.assertIn("s2", order.triggered_by)


if __name__ == "__main__":
    unittest.main()
