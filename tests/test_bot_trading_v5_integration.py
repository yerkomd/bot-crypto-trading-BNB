import os
import sys
import unittest
from pathlib import Path
import importlib.util

import pandas as pd


BOT_DIR = Path(__file__).resolve().parents[1]
BOT_PATH = BOT_DIR / "bot_trading_v5.py"


def load_bot_module(module_name: str = "bot_trading_v5"):
    root = str(BOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    spec = importlib.util.spec_from_file_location(module_name, BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DummyEvent:
    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, _timeout=None):
        self._set = True
        return True


class DummyRiskManager:
    def observe(self, *args, **kwargs):
        return None

    def can_open(self, *args, **kwargs):
        return True, "ok"

    def snapshot(self, *args, **kwargs):
        return {}

    def record_realized_pnl(self, *args, **kwargs):
        return None


class BotTradingV5IntegrationTests(unittest.TestCase):
    def test_build_v5_strategy_adapter_reads_env(self):
        old_path = os.environ.get("V5_MODEL_PATH")
        old_thr = os.environ.get("V5_THRESHOLD_OVERRIDE")
        try:
            os.environ["V5_MODEL_PATH"] = "/tmp/model.joblib"
            os.environ["V5_THRESHOLD_OVERRIDE"] = "0.77"
            bot = load_bot_module("bot_trading_v5_env_test")
            adapter = bot._build_v5_strategy_adapter()
            self.assertEqual(adapter.model_path, "/tmp/model.joblib")
            self.assertAlmostEqual(float(adapter.threshold_override), 0.77, places=8)
        finally:
            if old_path is None:
                os.environ.pop("V5_MODEL_PATH", None)
            else:
                os.environ["V5_MODEL_PATH"] = old_path
            if old_thr is None:
                os.environ.pop("V5_THRESHOLD_OVERRIDE", None)
            else:
                os.environ["V5_THRESHOLD_OVERRIDE"] = old_thr

    def test_run_strategy_opens_position_with_v5_risk_levels(self):
        bot = load_bot_module("bot_trading_v5_run_test")

        from backtesting.bt_types import EntrySignal

        class FakeStrategy:
            def prepare_indicators(self, *, symbol, df):
                out = df.copy()
                out["ema200"] = [90.0] * len(out)
                out["ema50"] = [95.0] * len(out)
                out["adx"] = [30.0] * len(out)
                out["atr"] = [2.0] * len(out)
                out["regime"] = ["BULL"] * len(out)
                return out

            def generate_entry(self, ctx):
                return EntrySignal(True, 0.05, None)

            def compute_risk_levels(self, *, symbol, regime, buy_price, indicators_row):
                return 110.0, 95.0, {
                    "tp_initial": 110.0,
                    "trailing_active": False,
                    "max_price": float(buy_price),
                    "atr_entry": 2.0,
                    "trailing_sl_atr_mult": 1.2,
                }

        old_stop = bot.STOP_EVENT
        old_require = bot._require_clients
        old_filters = bot.get_symbol_filters
        old_build = bot._build_v5_strategy_adapter
        old_get_balance = bot.get_balance
        old_get_data = bot.get_data_binance
        old_load = bot.load_positions
        old_save = bot.save_positions
        old_qty = bot.preparar_cantidad
        old_order = bot.ejecutar_orden_con_confirmacion
        old_send = bot.send_event_to_telegram
        old_log = bot.log_trade
        old_equity = bot.compute_equity_total
        old_risk = bot.RISK_MANAGER

        positions_store = []

        try:
            bot.STOP_EVENT = DummyEvent()
            bot._require_clients = lambda: None
            bot.get_symbol_filters = lambda symbol: {
                "min_qty": 0.000001,
                "step_size": 0.000001,
                "max_qty": 9999999.0,
                "min_notional": 0.0,
            }
            bot._build_v5_strategy_adapter = lambda: FakeStrategy()
            bot.get_balance = lambda: 1000.0
            bot.get_data_binance = lambda symbol, interval="1h", limit=260: pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
                    "open": [100.0, 100.0, 100.0],
                    "high": [101.0, 101.0, 101.0],
                    "low": [99.0, 99.0, 99.0],
                    "close": [100.0, 100.0, 100.0],
                    "volume": [1.0, 1.0, 1.0],
                }
            )
            bot.load_positions = lambda symbol: list(positions_store)

            def _save_positions(symbol, rows):
                positions_store.clear()
                positions_store.extend(list(rows or []))

            bot.save_positions = _save_positions
            bot.preparar_cantidad = lambda symbol, usd_balance_frac, price, filters=None: (0.5, None)
            bot.ejecutar_orden_con_confirmacion = lambda tipo, symbol, cantidad: {
                "executedQty": "0.5",
                "cummulativeQuoteQty": "50",
            }
            bot.send_event_to_telegram = lambda *args, **kwargs: None
            bot.log_trade = lambda *args, **kwargs: None
            bot.compute_equity_total = lambda *args, **kwargs: 1000.0
            bot.RISK_MANAGER = DummyRiskManager()

            import threading

            bot.run_strategy("BTCUSDT", threading.Lock())

            self.assertEqual(len(positions_store), 1)
            p = positions_store[0]
            self.assertAlmostEqual(float(p["take_profit"]), 110.0, places=8)
            self.assertAlmostEqual(float(p["stop_loss"]), 95.0, places=8)
            self.assertAlmostEqual(float(p["tp_initial"]), 110.0, places=8)
            self.assertEqual(bool(p["trailing_active"]), False)
            self.assertAlmostEqual(float(p["trailing_sl_atr_mult"]), 1.2, places=8)
            self.assertEqual(str(p["regime"]), "BULL")
        finally:
            bot.STOP_EVENT = old_stop
            bot._require_clients = old_require
            bot.get_symbol_filters = old_filters
            bot._build_v5_strategy_adapter = old_build
            bot.get_balance = old_get_balance
            bot.get_data_binance = old_get_data
            bot.load_positions = old_load
            bot.save_positions = old_save
            bot.preparar_cantidad = old_qty
            bot.ejecutar_orden_con_confirmacion = old_order
            bot.send_event_to_telegram = old_send
            bot.log_trade = old_log
            bot.compute_equity_total = old_equity
            bot.RISK_MANAGER = old_risk


if __name__ == "__main__":
    unittest.main()
