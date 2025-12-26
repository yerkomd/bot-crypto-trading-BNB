import csv
import tempfile
import unittest
from pathlib import Path
import importlib.util

import pandas as pd


BOT_DIR = Path(__file__).resolve().parents[1]
BOT_PATH = BOT_DIR / "bot_trading_v2_2.py"


def load_bot_module():
    spec = importlib.util.spec_from_file_location("bot_trading_v2_2", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DummyEvent:
    def __init__(self):
        self._set = False
        self.wait_calls = 0

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        # After first wait, stop loops
        self.wait_calls += 1
        self._set = True
        return True


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot_module()

    def test_detect_market_regime_bull_bear_lateral(self):
        bot = self.bot

        df_bull = pd.DataFrame(
            {
                "close": [101.0],
                "ema200": [100.0],
                "adx": [25.0],
            }
        )
        self.assertEqual(bot.detect_market_regime(df_bull), "BULL")

        df_bear = pd.DataFrame(
            {
                "close": [99.0],
                "ema200": [100.0],
                "adx": [30.0],
            }
        )
        self.assertEqual(bot.detect_market_regime(df_bear), "BEAR")

        df_lat = pd.DataFrame(
            {
                "close": [101.0],
                "ema200": [100.0],
                "adx": [10.0],
            }
        )
        self.assertEqual(bot.detect_market_regime(df_lat), "LATERAL")

        df_nan = pd.DataFrame(
            {
                "close": [101.0],
                "ema200": [pd.NA],
                "adx": [25.0],
            }
        )
        self.assertEqual(bot.detect_market_regime(df_nan), "LATERAL")

    def test_log_trade_writes_new_columns(self):
        bot = self.bot

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_csv = td_path / "trading_log_TEST.csv"

            original_get_log_filename = bot.get_log_filename
            bot.get_log_filename = lambda symbol: str(out_csv)
            try:
                bot.log_trade(
                    pd.Timestamp("2025-12-26"),
                    "TEST",
                    "buy",
                    100.0,
                    1.0,
                    0.0,
                    0.0,
                    30.0,
                    50.0,
                    40.0,
                    "BUY",
                    extra={
                        "regime": "BULL",
                        "rsi_threshold_used": 45,
                        "take_profit_pct_used": 5,
                        "stop_loss_pct_used": 2,
                        "trailing_tp_pct_used": 1.2,
                        "trailing_sl_pct_used": 1.0,
                        "buy_cooldown_used": 7200,
                        "position_size_used": 0.04,
                    },
                )

                with open(out_csv, "r", newline="") as rf:
                    reader = csv.reader(rf)
                    header = next(reader)

                # Verify new fields are present
                self.assertIn("regime", header)
                self.assertIn("rsi_threshold_used", header)
                self.assertIn("take_profit_pct_used", header)
                self.assertIn("stop_loss_pct_used", header)
                self.assertIn("trailing_tp_pct_used", header)
                self.assertIn("trailing_sl_pct_used", header)
                self.assertIn("buy_cooldown_used", header)
                self.assertIn("position_size_used", header)
            finally:
                bot.get_log_filename = original_get_log_filename

    def test_run_strategy_applies_regime_params_only_to_new_positions(self):
        bot = self.bot

        # Patch STOP_EVENT to end after one loop
        original_stop_event = bot.STOP_EVENT
        bot.STOP_EVENT = DummyEvent()

        # Make _require_client a no-op
        original_require_client = bot._require_client
        bot._require_client = lambda: None

        # Patches to avoid network
        original_get_balance = bot.get_balance
        bot.get_balance = lambda: 1000.0

        original_get_symbol_filters = bot.get_symbol_filters
        bot.get_symbol_filters = lambda symbol: {
            "min_notional": 10.0,
            "min_qty": 0.0,
            "step_size": 0.000001,
            "max_qty": 9999999.0,
        }

        original_get_data_binance = bot.get_data_binance
        bot.get_data_binance = lambda symbol, interval="1h", limit=260: pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-12-26T00:00:00", "2025-12-26T01:00:00", "2025-12-26T02:00:00"]
                ),
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )

        # Force BULL regime for this test
        original_detect_market_regime = bot.detect_market_regime
        bot.detect_market_regime = lambda df: "BULL"

        original_cal_metrics = bot.cal_metrics_technig

        def fake_cal_metrics(df, *_args, **_kwargs):
            # Ensure required indicators exist for the logic
            df = df.copy()
            df["rsi"] = [10.0, 10.0, 10.0]
            df["stochrsi_k"] = [2.0, 2.0, 2.0]
            df["stochrsi_d"] = [1.0, 1.0, 1.0]
            df["ema200"] = [90.0, 90.0, 90.0]
            df["adx"] = [30.0, 30.0, 30.0]
            return df

        bot.cal_metrics_technig = fake_cal_metrics

        original_preparar_cantidad = bot.preparar_cantidad
        bot.preparar_cantidad = lambda symbol, usd_balance_frac, price, filters=None: (0.01, None)

        original_debug_symbol_filters = bot.debug_symbol_filters
        bot.debug_symbol_filters = lambda symbol: None

        original_ejecutar = bot.ejecutar_orden_con_confirmacion
        bot.ejecutar_orden_con_confirmacion = lambda *args, **kwargs: {"status": "FILLED"}

        original_market_change = bot.market_change_last_5_intervals
        bot.market_change_last_5_intervals = lambda symbol: 0.0

        original_send_event = bot.send_event_to_telegram
        bot.send_event_to_telegram = lambda *args, **kwargs: None

        calls = {"log_trade": []}
        original_log_trade = bot.log_trade

        def capture_log_trade(*args, **kwargs):
            calls["log_trade"].append((args, kwargs))

        bot.log_trade = capture_log_trade

        positions_store = []
        original_load_positions = bot.load_positions
        original_save_positions = bot.save_positions

        bot.load_positions = lambda symbol: list(positions_store)

        def fake_save_positions(symbol, positions_list):
            positions_store.clear()
            positions_store.extend(positions_list)

        bot.save_positions = fake_save_positions

        lock = __import__("threading").Lock()

        try:
            bot.run_strategy("TESTUSDT", lock)

            self.assertEqual(len(positions_store), 1)
            pos = positions_store[0]

            # Assert position was created with BULL params
            self.assertEqual(pos.get("regime"), "BULL")
            self.assertEqual(pos.get("rsi_threshold"), bot.BULL["RSI_THRESHOLD"])
            self.assertEqual(pos.get("trailing_tp_pct"), bot.BULL["TRAILING_TP_PCT"])
            self.assertEqual(pos.get("trailing_sl_pct"), bot.BULL["TRAILING_SL_PCT"])

            # TP/SL set at entry using active params
            expected_tp = 100.0 * (1 + bot.BULL["TAKE_PROFIT_PCT"] / 100)
            expected_sl = 100.0 * (1 - bot.BULL["STOP_LOSS_PCT"] / 100)
            self.assertAlmostEqual(pos["take_profit"], expected_tp)
            self.assertAlmostEqual(pos["stop_loss"], expected_sl)

            # Verify trade log was called with extra parameters
            self.assertTrue(calls["log_trade"], "Expected log_trade to be called")
            last_call_args, last_call_kwargs = calls["log_trade"][-1]
            extra = last_call_kwargs.get("extra")
            self.assertIsInstance(extra, dict)
            self.assertEqual(extra.get("regime"), "BULL")
            self.assertEqual(extra.get("rsi_threshold_used"), float(bot.BULL["RSI_THRESHOLD"]))
            self.assertEqual(extra.get("take_profit_pct_used"), float(bot.BULL["TAKE_PROFIT_PCT"]))
            self.assertEqual(extra.get("stop_loss_pct_used"), float(bot.BULL["STOP_LOSS_PCT"]))
            self.assertEqual(extra.get("trailing_tp_pct_used"), float(bot.BULL["TRAILING_TP_PCT"]))
            self.assertEqual(extra.get("trailing_sl_pct_used"), float(bot.BULL["TRAILING_SL_PCT"]))
            self.assertEqual(extra.get("buy_cooldown_used"), int(bot.BULL["BUY_COOLDOWN"]))
            self.assertEqual(extra.get("position_size_used"), float(bot.BULL["POSITION_SIZE"]))

        finally:
            bot.STOP_EVENT = original_stop_event
            bot._require_client = original_require_client
            bot.get_balance = original_get_balance
            bot.get_symbol_filters = original_get_symbol_filters
            bot.get_data_binance = original_get_data_binance
            bot.detect_market_regime = original_detect_market_regime
            bot.cal_metrics_technig = original_cal_metrics
            bot.preparar_cantidad = original_preparar_cantidad
            bot.debug_symbol_filters = original_debug_symbol_filters
            bot.ejecutar_orden_con_confirmacion = original_ejecutar
            bot.market_change_last_5_intervals = original_market_change
            bot.send_event_to_telegram = original_send_event
            bot.log_trade = original_log_trade
            bot.load_positions = original_load_positions
            bot.save_positions = original_save_positions

    def test_monitoring_open_position_uses_trailing_per_position(self):
        bot = self.bot

        original_stop_event = bot.STOP_EVENT
        bot.STOP_EVENT = DummyEvent()

        original_require_client = bot._require_client
        bot._require_client = lambda: None

        # Prepare a position that will trigger trailing update
        positions_store = [
            {
                "buy_price": 100.0,
                "amount": 0.01,
                "timestamp": "2025-12-26T00:00:00",
                "take_profit": 110.0,
                "stop_loss": 90.0,
                "trailing_tp_pct": 1.2,
                "trailing_sl_pct": 1.0,
            }
        ]

        original_load_positions = bot.load_positions
        original_save_positions = bot.save_positions

        bot.load_positions = lambda symbol: list(positions_store)

        def fake_save_positions(symbol, positions_list):
            positions_store.clear()
            positions_store.extend(positions_list)

        bot.save_positions = fake_save_positions

        # Price jumps above take_profit to trigger trailing
        original_get_data_binance = bot.get_data_binance
        bot.get_data_binance = lambda symbol, interval="1h", limit=40: pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-12-26T00:00:00"]),
                "open": [120.0],
                "high": [121.0],
                "low": [119.0],
                "close": [120.0],
                "volume": [1.0],
            }
        )

        original_cal_metrics = bot.cal_metrics_technig
        bot.cal_metrics_technig = lambda df, *_args, **_kwargs: df

        # Avoid sells
        original_get_free_base_asset = bot.get_free_base_asset
        bot.get_free_base_asset = lambda symbol: 0.0

        original_sanitize_quantity = bot.sanitize_quantity
        bot.sanitize_quantity = lambda *args, **kwargs: (None, "skip")

        original_ejecutar = bot.ejecutar_orden_con_confirmacion
        bot.ejecutar_orden_con_confirmacion = lambda *args, **kwargs: None

        original_send_event = bot.send_event_to_telegram
        bot.send_event_to_telegram = lambda *args, **kwargs: None

        original_log_trade = bot.log_trade
        bot.log_trade = lambda *args, **kwargs: None

        lock = __import__("threading").Lock()

        try:
            bot.monitoring_open_position("TESTUSDT", lock)

            self.assertEqual(len(positions_store), 1)
            pos = positions_store[0]

            # Updated TP/SL should use per-position trailing percentages
            expected_tp = 120.0 * (1 + 1.2 / 100)
            expected_sl = 120.0 * (1 - 1.0 / 100)
            self.assertAlmostEqual(pos["take_profit"], expected_tp)
            self.assertAlmostEqual(pos["stop_loss"], expected_sl)

        finally:
            bot.STOP_EVENT = original_stop_event
            bot._require_client = original_require_client
            bot.load_positions = original_load_positions
            bot.save_positions = original_save_positions
            bot.get_data_binance = original_get_data_binance
            bot.cal_metrics_technig = original_cal_metrics
            bot.get_free_base_asset = original_get_free_base_asset
            bot.sanitize_quantity = original_sanitize_quantity
            bot.ejecutar_orden_con_confirmacion = original_ejecutar
            bot.send_event_to_telegram = original_send_event
            bot.log_trade = original_log_trade


if __name__ == "__main__":
    unittest.main(verbosity=2)
