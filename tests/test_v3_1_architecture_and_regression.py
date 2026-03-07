import os
import sys
import unittest
from pathlib import Path
import importlib.util
import warnings

import pandas as pd


BOT_DIR = Path(__file__).resolve().parents[1]
BOT_PATH = BOT_DIR / "bot_trading_v3_1.py"


def load_bot_module():
    # Allow running this test file directly from ./tests (python test_*.py)
    # by ensuring the project root is importable (so `import db...` works).
    root = str(BOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Third-party noise: python-binance currently triggers websockets deprecation warnings.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"binance\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\..*")

    spec = importlib.util.spec_from_file_location("bot_trading_v3_1", BOT_PATH)
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
        # Stop loops quickly in unit tests
        self.wait_calls += 1
        self._set = True
        return True


class FakeCursor:
    def __init__(self, existing_ids=None):
        self._existing_ids = existing_ids or []
        self._last_select_existing = False

    def execute(self, sql, params=None):
        s = str(sql).strip().lower()
        # Used by replace_positions(): SELECT id FROM trading.open_positions WHERE symbol = %s
        self._last_select_existing = s.startswith("select id from trading.open_positions")

    def fetchall(self):
        if self._last_select_existing:
            return [{"id": x} for x in self._existing_ids]
        return []


class FakeDB:
    def __init__(self, cursor=None):
        self._cursor = cursor or FakeCursor()

    def run(self, fn, **_kwargs):
        return fn(self._cursor)


class ArchitectureAndRegressionV31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot_module()

    def test_get_data_binance_reads_postgres_via_read_klines_df(self):
        bot = self.bot

        calls = []

        def fake_read_klines_df(*, db, symbol, interval, limit):
            calls.append({"db": db, "symbol": symbol, "interval": interval, "limit": limit})
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-02-12T00:00:00"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            )

        original_db = bot.DB
        original_read = bot.read_klines_df
        try:
            bot.DB = object()
            bot.read_klines_df = fake_read_klines_df

            df = bot.get_data_binance("btcusdt", interval="4h", limit=123)
            self.assertFalse(df.empty)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["db"], bot.DB)
            self.assertEqual(calls[0]["symbol"], "BTCUSDT")
            self.assertEqual(calls[0]["interval"], "4h")
            self.assertEqual(calls[0]["limit"], 123)
        finally:
            bot.DB = original_db
            bot.read_klines_df = original_read

    def test_log_trade_goes_to_trade_repo_and_normalizes_reason(self):
        bot = self.bot

        inserted = {}

        class DummyTradeRepo:
            def insert_trade(self, **kwargs):
                inserted.update(kwargs)
                return 1

        # log_trade requires repos initialized
        original_trade = bot.TRADE_REPO
        original_open = bot.OPEN_POS_REPO
        original_equity = bot.EQUITY_REPO
        try:
            bot.TRADE_REPO = DummyTradeRepo()
            bot.OPEN_POS_REPO = object()
            bot.EQUITY_REPO = object()

            bot.log_trade(
                pd.Timestamp("2026-02-12T00:00:00"),
                "btcusdt",
                "sell",
                100.0,
                0.01,
                1.23,
                0.5,
                40.0,
                10.0,
                20.0,
                "tp",  # should normalize to TAKE_PROFIT
                extra={"buy_price": 90.0, "regime": "BULL"},
            )

            self.assertEqual(inserted.get("symbol"), "BTCUSDT")
            self.assertEqual(inserted.get("side"), "SELL")
            self.assertEqual(inserted.get("reason"), "TAKE_PROFIT")
            self.assertEqual(inserted.get("buy_price"), 90.0)
            self.assertEqual(inserted.get("sell_price"), 100.0)
            self.assertEqual(inserted.get("regime"), "BULL")
        finally:
            bot.TRADE_REPO = original_trade
            bot.OPEN_POS_REPO = original_open
            bot.EQUITY_REPO = original_equity

    def test_open_positions_replace_positions_persists_inserted_id_back(self):
        # Unit test for repository bugfix: inserted rows must write id back to the dict
        from repositories.open_positions_repo import OpenPositionsRepository

        cursor = FakeCursor(existing_ids=[])
        db = FakeDB(cursor=cursor)
        repo = OpenPositionsRepository(db)

        original_insert = repo._insert_in_cursor
        try:
            repo._insert_in_cursor = lambda cur, pos: 123
            positions = [{"buy_price": 100.0, "amount": 0.01, "timestamp": "2026-02-12T00:00:00"}]
            ok = repo.replace_positions("BTCUSDT", positions)
            self.assertTrue(ok)
            self.assertEqual(positions[0].get("id"), 123)
        finally:
            repo._insert_in_cursor = original_insert

    def test_monitoring_uses_binance_live_price_for_stop_loss(self):
        bot = self.bot

        # Stop after one loop
        original_stop_event = bot.STOP_EVENT
        bot.STOP_EVENT = DummyEvent()

        # Avoid requiring real clients
        original_require = bot._require_clients
        bot._require_clients = lambda: None

        # Avoid network / external dependencies
        original_get_symbol_filters = bot.get_symbol_filters
        bot.get_symbol_filters = lambda symbol: {
            "min_qty": 0.000001,
            "step_size": 0.000001,
            "max_qty": 9999999.0,
            "min_notional": 0.0,
        }

        # Position will stop out at 90; DB close is 100 but live price is 80
        positions_store = [
            {
                "id": 1,
                "buy_price": 100.0,
                "amount": 0.01,
                "timestamp": "2026-02-12T00:00:00",
                "take_profit": 120.0,
                "stop_loss": 90.0,
                "trailing_active": False,
                "max_price": 100.0,
            }
        ]

        original_load_positions = bot.load_positions
        original_save_positions = bot.save_positions

        bot.load_positions = lambda symbol: list(positions_store)

        def fake_save_positions(symbol, positions_list):
            positions_store.clear()
            positions_store.extend(list(positions_list or []))

        bot.save_positions = fake_save_positions

        original_get_data = bot.get_data_binance
        bot.get_data_binance = lambda symbol, interval="1h", limit=40: pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-02-12T00:00:00", "2026-02-12T01:00:00"]),
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1.0, 1.0],
            }
        )

        original_cal_metrics = bot.cal_metrics_technig

        def fake_cal_metrics(df, *_args, **_kwargs):
            df = df.copy()
            df["rsi"] = [50.0] * len(df)
            df["stochrsi_k"] = [50.0] * len(df)
            df["stochrsi_d"] = [50.0] * len(df)
            # ATR is used with [-2] index; ensure >= 2 rows
            df["atr"] = [1.0] * len(df)
            return df

        bot.cal_metrics_technig = fake_cal_metrics

        original_get_live = bot.get_precio_actual
        bot.get_precio_actual = lambda symbol: 80.0

        original_get_balance = bot.get_balance
        bot.get_balance = lambda: 1000.0

        original_get_base_bal = bot.get_base_asset_balance
        bot.get_base_asset_balance = lambda symbol: {"free": 0.01, "locked": 0.0}

        original_sanitize = bot.sanitize_quantity
        bot.sanitize_quantity = lambda symbol, qty, price, for_sell=False, filters=None: (qty, None)

        original_cancel_all = bot.cancel_all_open_orders
        bot.cancel_all_open_orders = lambda symbol: 0

        original_send_event = bot.send_event_to_telegram
        bot.send_event_to_telegram = lambda *args, **kwargs: None

        original_exec_order = bot.ejecutar_orden_con_confirmacion
        bot.ejecutar_orden_con_confirmacion = lambda *args, **kwargs: {"status": "FILLED"}

        log_calls = []
        original_log_trade = bot.log_trade

        def capture_log_trade(*args, **kwargs):
            log_calls.append((args, kwargs))

        bot.log_trade = capture_log_trade

        lock = __import__("threading").Lock()

        try:
            bot.monitoring_open_position("BTCUSDT", lock)

            # Position should have been removed after stop-loss sell
            self.assertEqual(positions_store, [])

            # Ensure we logged a SELL using live price (80) not DB close (100)
            self.assertTrue(log_calls, "Expected log_trade to be called")
            args, _kwargs = log_calls[-1]
            # log_trade(timestamp, symbol, 'sell', price, ...)
            self.assertEqual(args[1], "BTCUSDT")
            self.assertEqual(str(args[2]).lower(), "sell")
            self.assertAlmostEqual(float(args[3]), 80.0)
        finally:
            bot.STOP_EVENT = original_stop_event
            bot._require_clients = original_require
            bot.get_symbol_filters = original_get_symbol_filters
            bot.load_positions = original_load_positions
            bot.save_positions = original_save_positions
            bot.get_data_binance = original_get_data
            bot.cal_metrics_technig = original_cal_metrics
            bot.get_precio_actual = original_get_live
            bot.get_balance = original_get_balance
            bot.get_base_asset_balance = original_get_base_bal
            bot.sanitize_quantity = original_sanitize
            bot.cancel_all_open_orders = original_cancel_all
            bot.send_event_to_telegram = original_send_event
            bot.ejecutar_orden_con_confirmacion = original_exec_order
            bot.log_trade = original_log_trade


if __name__ == "__main__":
    unittest.main()
