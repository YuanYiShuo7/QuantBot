from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from typing import Tuple

from account.base import Account
from market.base import Market
from exchange.base import Exchange
from reward.base import Reward
from llm_agent.base import LLMAgent

from general.interfaces.environment_interface import EnvironmentInterface

class Environment(EnvironmentInterface):
    """交易环境实现类 - 负责强化学习更新逻辑"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 环境配置参数
        self.trading_interval_days = self.config.get('trading_interval_days', 1)  # 交易间隔天数
        self.start_timestamp = self.config.get('start_timestamp')
        self.end_timestamp = self.config.get('end_timestamp')
        self.decision_time = self.config.get('decision_time', {'hour': 14, 'minute': 45})  # 决策时间
        
        # 状态跟踪
        self.current_timestamp = self.start_timestamp
        self.last_trading_day = None
        self.trading_day_count = 0
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("交易环境初始化完成")
        self.logger.info(f"交易间隔: {self.trading_interval_days} 天")
        self.logger.info(f"时间范围: {self.start_timestamp} 到 {self.end_timestamp}")
        self.logger.info(f"决策时间: {self.decision_time['hour']}:{self.decision_time['minute']}")
    
    def step(self, timestamp: datetime, account: Account, exchange: Exchange, 
            llm_agent: LLMAgent, market: Market, reward: Reward) -> Tuple[bool, datetime]:
        """
        执行单个交易时间步
        
        Args:
            timestamp: 当前时间戳
            account: 账户数据
            exchange: 模拟交易所接口
            llm_agent: LLM代理
            market: 模拟市场接口
            reward: 奖励系统
            
        Returns:
            bool: 是否达到终止条件
            datetime: 下一个时间戳（如果终止则返回当前时间戳）
        """
        try:
            self.logger.info(f"开始执行时间步: {timestamp}")
            
            # 1. 更新时间戳到下一个交易日（基于实际市场数据）
            next_timestamp = self._get_next_timestamp(market, timestamp)
            
            # 检查是否没有下一个交易日（数据结束）
            if next_timestamp is None:
                self.logger.info("没有更多交易日数据，环境运行结束")
                return True, timestamp
            
            self.current_timestamp = next_timestamp
            
            # 检查是否达到配置的终止条件
            if self._is_terminated():
                self.logger.info("达到配置的终止条件，环境运行结束")
                return True, next_timestamp
            
            # 2. 更新市场数据
            self.logger.info("更新市场数据...")
            market_schema = market.update_market_from_data_cache(next_timestamp)
            account_schema = account.account

            # 3. 检查是否到了交易决策日
            should_trade = self._should_trade_today(next_timestamp)
            
            if should_trade:
                self.logger.info("今日为交易决策日，执行Agent决策流程")
                # Agent生成动作
                prompt, output, actions = llm_agent.generate_pipeline(market, account)
                self.logger.info(f"Agent生成 {len(actions)} 个动作")
                
                reward.record_trajectory(next_timestamp, account_schema, prompt, output)
                
                # Exchange记录订单
                if actions:
                    self.logger.info("向交易所提交订单...")
                    exchange.update_orders(actions, account_schema, next_timestamp)
            
            # 4. Exchange执行订单并更新信息
            self.logger.info("执行订单检查...")
            order_results = exchange.check_and_execute_orders(market_schema, account_schema, next_timestamp)
            
            if order_results:
                for result in order_results:
                    self.logger.info(f"订单 {result.order_id} 执行结果: {result.status}")

            # 更新交易计数
            self.trading_day_count += 1
            self.last_trading_day = next_timestamp
            
            self.logger.info(f"时间步完成: {next_timestamp}, 交易天数: {self.trading_day_count}")
            return False, next_timestamp
            
        except Exception as e:
            self.logger.error(f"执行时间步失败: {str(e)}")
            return True, timestamp  # 发生错误时终止
        
    def _get_next_timestamp(self, market: Market, current_timestamp: datetime) -> Optional[datetime]:
        """
        获取下一个时间戳（基于市场数据的实际交易日）
        
        Args:
            market: 市场接口
            current_timestamp: 当前时间戳
            
        Returns:
            下一个交易日的时间戳，如果没有更多数据则返回None
        """
        try:
            # 使用market的方法获取下一个实际交易日
            next_timestamp = market.get_next_trading_day(current_timestamp)
            
            if next_timestamp is None:
                self.logger.warning("无法获取下一个交易日，可能已到达数据末尾")
                return None
                
            self.logger.debug(f"下一个交易日: {next_timestamp}")
            return next_timestamp
            
        except Exception as e:
            self.logger.error(f"获取下一个时间戳失败: {str(e)}")
            return None
        
    def _is_terminated(self) -> bool:
        """检查是否达到终止条件"""
        if not self.end_timestamp:
            return False
        
        return self.current_timestamp >= self.end_timestamp
    
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