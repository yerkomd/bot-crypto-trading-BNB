"""Backtesting package for the institutional bot.

Core entrypoint: `BacktestEngine`.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult

__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult"]
