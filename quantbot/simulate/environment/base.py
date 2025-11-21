from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from account.base import Account
from market.base import Market
from exchange.base import Exchange
from reward.base import Reward
from llm_agent.base import LLMAgent
from timer.base import Timer

from general.interfaces.environment_interface import EnvironmentInterface

class Environment(EnvironmentInterface):
    """交易环境实现类 - 负责强化学习更新逻辑"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 环境配置参数
        self.trading_interval_days = self.config.get('trading_interval_days', 1)  # 交易间隔天数
        self.should_trade = False
        self.last_trading_day = None  # 上次交易日
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("交易环境初始化完成")
        self.logger.info(f"交易间隔: {self.trading_interval_days} 天")
    
    def step(self, timer: Timer, account: Account, exchange: Exchange, 
            llm_agent: LLMAgent, market: Market, reward: Reward) -> bool:
        """
        执行单个交易时间步
        
        Args:
            timer: 计时器
            account: 账户数据
            exchange: 模拟交易所接口
            llm_agent: LLM代理
            market: 模拟市场接口
            reward: 奖励系统
            
        Returns:
            bool: 是否达到终止条件
        """
        try:   
            timestamp = timer.get_current_timestamp()
            self.logger.info(f"当前时间戳: {timestamp}")
            
            
            self.logger.info("更新市场数据...")
            market.update_market_from_data_cache(timestamp)

            self.should_trade = self._should_trade_today(timestamp)
            
            if self.should_trade:
                self.logger.info("今日为交易决策日，执行Agent决策流程")
                # Agent生成动作
                prompt, output, actions = llm_agent.generate_pipeline(account, market)
                self.logger.info(f"Agent Output: {output}")
                
                reward.record_trajectory(timestamp, account, prompt, output)
                self.last_trading_day = timestamp
                
                # Exchange记录订单
                if actions:
                    self.logger.info("向交易所提交订单...")
                    exchange.update_orders(actions, account, timestamp)
            
            self.logger.info("执行订单检查...")
            order_results = exchange.check_and_execute_orders(market, account, timestamp)
            
            if order_results:
                for result in order_results:
                    self.logger.info(f"订单 {result.order_id} 执行结果: {result.status}")

            self.logger.info(f"时间步完成: {timestamp}")

            if (timer.step(market)):
                self.logger.info("达到终止条件，结束交易环境步骤")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"执行时间步失败: {str(e)}")
            return True
    
    def _should_trade_today(self, timestamp: datetime) -> bool:
        """
        检查今天是否应该进行交易决策
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            bool: 是否应该交易
        """
        # 如果是第一个交易日，总是进行交易
        if self.last_trading_day is None:
            return True
        
        # 计算距离上次交易的天数
        days_since_last_trade = (timestamp.date() - self.last_trading_day.date()).days
        
        # 检查是否达到交易间隔
        return days_since_last_trade >= self.trading_interval_days