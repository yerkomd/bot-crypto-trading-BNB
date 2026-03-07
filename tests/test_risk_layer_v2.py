import sys
import unittest
from pathlib import Path


BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))


import risk_layer_v2 as rl2


class FakeCursor:
    def __init__(self):
        self.executed = []
        self._last_sql = None
        self._fetchone_queue = []

    def queue_fetchone(self, row):
        self._fetchone_queue.append(row)

    def execute(self, sql, params=None):
        self._last_sql = str(sql)
        self.executed.append((str(sql), params))

    def fetchone(self):
        if self._fetchone_queue:
            return self._fetchone_queue.pop(0)
        return None

    def fetchall(self):
        return []


class FakeDB:
    def __init__(self, cursor=None):
        self.cursor = cursor or FakeCursor()
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

    def wait(self, _timeout=None):
        return self._set


class RiskLayerV2UnitTests(unittest.TestCase):
    def test_compute_atr_simple_returns_none_when_insufficient_data(self):
        atr = rl2._compute_atr_simple(
            highs=[10.0, 11.0],
            lows=[9.0, 10.0],
            closes=[9.5, 10.5],
            period=14,
        )
        self.assertIsNone(atr)

    def test_compute_atr_simple_computes_positive_value(self):
        highs = [10, 11, 12, 13, 14, 15]
        lows = [9, 10, 11, 12, 13, 14]
        closes = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        atr = rl2._compute_atr_simple(highs=highs, lows=lows, closes=closes, period=3)
        self.assertIsNotNone(atr)
        self.assertGreater(float(atr), 0.0)

    def test_global_risk_controller_triggers_kill_switch_and_persists(self):
        cur = FakeCursor()
        db = FakeDB(cur)
        stop_event = DummyStopEvent()

        telegram_msgs = []

        def send_telegram(msg: str):
            telegram_msgs.append(msg)

        event_logger = rl2.RiskEventLogger(db)
        ctrl = rl2.GlobalRiskController(
            db=db,
            stop_event=stop_event,
            send_telegram=send_telegram,
            event_logger=event_logger,
            enabled=True,
            max_drawdown_frac=0.10,
        )

        # First snapshot sets peak
        ctrl.on_equity_snapshot(equity_total=1000.0)
        self.assertFalse(stop_event.is_set())

        # Drop 20% triggers kill-switch
        ctrl.on_equity_snapshot(equity_total=800.0)
        self.assertTrue(stop_event.is_set())
        self.assertTrue(any("GLOBAL KILL SWITCH" in m for m in telegram_msgs))

        # Ensure DB persistence happened: global_risk_state + risk_events inserts
        joined = "\n".join(s for (s, _p) in cur.executed)
        self.assertIn("INSERT INTO trading.global_risk_state", joined)
        self.assertIn("INSERT INTO trading.risk_events", joined)

    def test_system_health_monitor_circuit_breaker(self):
        stop_event = DummyStopEvent()
        msgs = []

        def send(msg: str):
            msgs.append(msg)

        db = FakeDB()
        event_logger = rl2.RiskEventLogger(db)
        mon = rl2.SystemHealthMonitor(
            stop_event=stop_event,
            send_telegram=send,
            event_logger=event_logger,
            max_critical_errors=2,
        )

        mon.record_critical(reason="X")
        self.assertFalse(stop_event.is_set())
        mon.record_critical(reason="X")
        self.assertTrue(stop_event.is_set())
        self.assertTrue(any("CIRCUIT BREAKER" in m for m in msgs))

        # Reset on success
        stop_event2 = DummyStopEvent()
        mon2 = rl2.SystemHealthMonitor(
            stop_event=stop_event2,
            send_telegram=lambda _m: None,
            event_logger=event_logger,
            max_critical_errors=3,
        )
        mon2.record_critical(reason="A")
        mon2.record_success()
        self.assertEqual(mon2.critical_errors_consecutive, 0)

    def test_reconcile_case_a_db_has_position_exchange_zero_deletes_db_and_logs(self):
        cur = FakeCursor()
        db = FakeDB(cur)
        event_logger = rl2.RiskEventLogger(db)

        class FakeOpenRepo:
            def __init__(self):
                self.replaced = []

            def list_by_symbol(self, symbol):
                return [{"id": 1, "symbol": symbol, "buy_price": 100.0, "amount": 0.5, "opened_at": "x"}]

            def replace_positions(self, symbol, positions):
                self.replaced.append((symbol, positions))
                return True

            def insert(self, position):
                raise AssertionError("Should not insert in case A")

        class FakeClient:
            def get_asset_balance(self, asset):
                return {"free": "0"}

        repo = FakeOpenRepo()
        rl2.reconcile_positions_with_exchange(
            symbol="BTCUSDT",
            db=db,
            open_pos_repo=repo,
            client_trade=FakeClient(),
            get_live_price=lambda _s: 123.0,
            event_logger=event_logger,
        )

        self.assertEqual(repo.replaced, [("BTCUSDT", [])])
        joined = "\n".join(s for (s, _p) in cur.executed)
        self.assertIn("INSERT INTO trading.reconciliation_events", joined)
        self.assertIn("INSERT INTO trading.risk_events", joined)

    def test_reconcile_case_b_exchange_has_asset_db_empty_inserts_recovered(self):
        cur = FakeCursor()
        db = FakeDB(cur)
        event_logger = rl2.RiskEventLogger(db)

        class FakeOpenRepo:
            def __init__(self):
                self.inserted = []

            def list_by_symbol(self, symbol):
                return []

            def replace_positions(self, symbol, positions):
                raise AssertionError("Should not replace in case B")

            def insert(self, position):
                self.inserted.append(position)
                return 99

        class FakeClient:
            def get_asset_balance(self, asset):
                return {"free": "0.25"}

        repo = FakeOpenRepo()
        rl2.reconcile_positions_with_exchange(
            symbol="BTCUSDT",
            db=db,
            open_pos_repo=repo,
            client_trade=FakeClient(),
            get_live_price=lambda _s: 200.0,
            event_logger=event_logger,
        )

        self.assertEqual(len(repo.inserted), 1)
        self.assertEqual(repo.inserted[0].get("regime"), "RECOVERED")
        self.assertAlmostEqual(float(repo.inserted[0].get("amount")), 0.25)

        # Never leave RECOVERED positions unmanaged
        bp = float(repo.inserted[0].get("buy_price"))
        sl = repo.inserted[0].get("stop_loss")
        tp = repo.inserted[0].get("take_profit")
        self.assertIsNotNone(sl)
        self.assertIsNotNone(tp)
        self.assertLess(float(sl), bp)
        self.assertGreater(float(tp), bp)

        joined = "\n".join(s for (s, _p) in cur.executed)
        self.assertIn("INSERT INTO trading.reconciliation_events", joined)
        self.assertIn("INSERT INTO trading.risk_events", joined)

    def test_reconcile_case_c_amount_mismatch_adjusts_db(self):
        cur = FakeCursor()
        db = FakeDB(cur)
        event_logger = rl2.RiskEventLogger(db)

        class FakeOpenRepo:
            def __init__(self):
                self.replaced = []

            def list_by_symbol(self, symbol):
                return [
                    {"id": 1, "symbol": symbol, "buy_price": 100.0, "amount": 0.6, "opened_at": "x"},
                    {"id": 2, "symbol": symbol, "buy_price": 110.0, "amount": 0.4, "opened_at": "x"},
                ]

            def replace_positions(self, symbol, positions):
                self.replaced.append((symbol, positions))
                return True

            def insert(self, position):
                raise AssertionError("Should not insert in case C")

        class FakeClient:
            def get_asset_balance(self, asset):
                return {"free": "0.5"}

        repo = FakeOpenRepo()
        rl2.reconcile_positions_with_exchange(
            symbol="BTCUSDT",
            db=db,
            open_pos_repo=repo,
            client_trade=FakeClient(),
            get_live_price=lambda _s: 123.0,
            event_logger=event_logger,
        )

        self.assertEqual(len(repo.replaced), 1)
        _sym, new_positions = repo.replaced[0]
        new_total = sum(float(p.get("amount") or 0.0) for p in new_positions)
        self.assertAlmostEqual(new_total, 0.5)
        joined = "\n".join(s for (s, _p) in cur.executed)
        self.assertIn("INSERT INTO trading.reconciliation_events", joined)
        self.assertIn("INSERT INTO trading.risk_events", joined)

    def test_upsert_risk_metrics_daily_executes_upsert(self):
        cur = FakeCursor()
        db = FakeDB(cur)

        rl2.upsert_risk_metrics_daily(
            db,
            day=rl2.date(2026, 1, 1),
            equity_open=1000.0,
            equity_close=1020.0,
            daily_return_pct=0.02,
            max_drawdown_intraday=0.01,
            realized_pnl=10.0,
            floating_pnl=5.0,
            total_exposure_frac=0.2,
        )

        self.assertGreaterEqual(len(cur.executed), 1)
        sql = cur.executed[-1][0]
        self.assertIn("INSERT INTO trading.risk_metrics_daily", sql)
        self.assertIn("ON CONFLICT (date)", sql)


class RiskLayerV2IntegrationTests(unittest.TestCase):
    def test_bot_binance_call_records_critical_on_minus_2015(self):
        # Minimal integration: ensure bot._binance_call triggers SystemHealthMonitor on -2015.
        import importlib.util

        bot_path = BOT_DIR / "bot_trading_v3_1.py"
        spec = importlib.util.spec_from_file_location("bot_trading_v3_1", bot_path)
        bot = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(bot)

        bot.client_data = object()
        bot.client_trade = object()

        stop_event = DummyStopEvent()
        mon = rl2.SystemHealthMonitor(
            stop_event=stop_event,
            send_telegram=lambda _m: None,
            event_logger=rl2.RiskEventLogger(FakeDB()),
            max_critical_errors=1,
        )
        bot.SYSTEM_HEALTH_MONITOR = mon

        class MyBinanceException(bot.BinanceAPIException):
            def __init__(self):
                self.code = -2015
                self.message = "Invalid API-key, IP, or permissions for action"

        def failing():
            raise MyBinanceException()

        with self.assertRaises(bot.BinanceAPIException):
            bot._binance_call(failing, tries=1)

        self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
