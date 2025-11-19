from typing import Dict, Any
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from general.interfaces.simulator_interface import SimulatorInterface


class Simulator(SimulatorInterface):
    """单轮交易模拟器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.components = self._initialize_components(self.config)
        self.environment = self.components['environment']
        self.timer = self.components['timer']
        self.account = self.components['account']
        self.exchange = self.components['exchange']
        self.market = self.components['market']
        self.llm_agent = self.components['llm_agent']
        self.reward = self.components['reward']

        self.logger.info("模拟器初始化完成，所有组件已就绪")

    def run(self) -> None:
        """运行单轮模拟：直到环境结束，再计算与持久化奖励"""
        try:
            step_count = 0
            finished = False

            while not finished:
                step_count += 1
                self.logger.debug("开始执行第 %d 个时间步", step_count)

                finished = self.environment.step(
                    self.timer,
                    self.account,
                    self.exchange,
                    self.llm_agent,
                    self.market,
                    self.reward,
                )

                if step_count > 1000:
                    self.logger.warning("达到最大步数限制(1000)，提前终止模拟")
                    break

            self.logger.info("环境运行结束，共执行 %d 个时间步", step_count)

            self.reward.calculate_score()
            self.reward.persist_data()

        except Exception as exc:
            self.logger.error("模拟器运行失败: %s", exc)
            raise

    def _initialize_components(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """根据配置初始化环境所需组件"""
        try:
            from environment.base import Environment
            from timer.base import Timer
            from account.base import Account
            from exchange.base import Exchange
            from market.base import Market
            from llm_agent.base import LLMAgent
            from reward.base import Reward

            components = {
                "environment": Environment(config_data.get("environment", {})),
                "timer": Timer(config_data.get("timer", {})),
                "account": Account(config_data.get("account", {})),
                "exchange": Exchange(config_data.get("exchange", {})),
                "market": Market(config_data.get("market", {})),
                "llm_agent": LLMAgent(config_data.get("llm_agent", {})),
                "reward": Reward(config_data.get("reward", {})),
            }

            self.logger.info("组件初始化完成: %s", ", ".join(components.keys()))
            return components

        except Exception as exc:
            self.logger.error("初始化组件失败: %s", exc)
            raise