import os
import sys
import unittest

# Permite importar el módulo desde la raíz del proyecto del bot
THIS_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

import bot_trading_v2 as bot


class TestQuantityUtils(unittest.TestCase):
    def test_floor_to_step_basic(self):
        self.assertAlmostEqual(bot.floor_to_step(1.234, 0.01), 1.23)

    def test_floor_to_step_smaller_than_step(self):
        self.assertEqual(bot.floor_to_step(0.009, 0.01), 0.0)

    def test_sanitize_quantity_ok(self):
        filters = {
            'min_qty': 0.001,
            'step_size': 0.0001,
            'max_qty': 999999.0,
            'min_notional': 10.0,
        }
        qty, motivo = bot.sanitize_quantity("BTCUSDT", 0.001234, 10000.0, filters=filters)
        self.assertIsNone(motivo)
        self.assertAlmostEqual(qty, 0.0012)

    def test_sanitize_quantity_min_qty(self):
        filters = {
            'min_qty': 0.01,
            'step_size': 0.001,
            'max_qty': 999999.0,
            'min_notional': 10.0,
        }
        qty, motivo = bot.sanitize_quantity("BTCUSDT", 0.005, 1000.0, filters=filters)
        self.assertIsNone(qty)
        self.assertIn("qty<", motivo)

    def test_sanitize_quantity_min_notional(self):
        filters = {
            'min_qty': 0.001,
            'step_size': 0.0001,
            'max_qty': 999999.0,
            'min_notional': 10.0,
        }
        qty, motivo = bot.sanitize_quantity("BTCUSDT", 0.001, 5000.0, filters=filters)  # notional=5
        self.assertIsNone(qty)
        self.assertIn("notional<", motivo)

    def test_preparar_cantidad_ok(self):
        filters = {
            'min_qty': 0.01,
            'step_size': 0.01,
            'max_qty': 999999.0,
            'min_notional': 10.0,
        }
        qty, motivo = bot.preparar_cantidad("FOOUSDT", 50.0, 100.0, filters=filters)  # raw=0.5
        self.assertIsNone(motivo)
        self.assertAlmostEqual(qty, 0.5)


if __name__ == "__main__":
    unittest.main()
