"""Strategy adapters for the backtesting engine."""

from .bot_v2_2_strategy import BotV2_2StrategyAdapter
from .bot_v3_1_strategy import BotV3_1StrategyAdapter
from .bot_v4_strategy import BotV4StrategyAdapter
from .bot_v5_strategy import BotV5StrategyAdapter

__all__ = [
	"BotV2_2StrategyAdapter",
	"BotV3_1StrategyAdapter",
	"BotV4StrategyAdapter",
	"BotV5StrategyAdapter",
]
