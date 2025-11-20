from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import logging


from general.interfaces.timer_interface import TimerInterface
from market.base import Market

class Timer(TimerInterface):
    """模拟交易计时器实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 从配置获取时间参数
        self.start_timestamp = self.config.get('start_timestamp')
        self.current_timestamp = self.config.get('current_timestamp', self.start_timestamp)
        self.end_timestamp = self.config.get('end_timestamp')
        
        # 验证必要参数
        if self.start_timestamp is None:
            raise ValueError("必须提供 start_timestamp 参数")
        if self.end_timestamp is None:
            raise ValueError("必须提供 end_timestamp 参数")
        
        # 确保当前时间戳在有效范围内
        if self.current_timestamp is None:
            self.current_timestamp = self.start_timestamp
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"计时器初始化完成")
        self.logger.info(f"开始时间: {self.start_timestamp}")
        self.logger.info(f"当前时间: {self.current_timestamp}")
        self.logger.info(f"结束时间: {self.end_timestamp}")
    
    def get_current_timestamp(self) -> datetime:
        """获取当前时间戳"""
        return self.current_timestamp

    def set_current_timestamp(self, current_timestamp: datetime):
        """设置当前时间戳"""
        self.current_timestamp = current_timestamp
        self.logger.debug(f"设置当前时间戳: {current_timestamp}")

    def step(self, market: Market) -> bool:
        """
        推进时间步长，返回更新后的当前时间戳
        
        Args:
            market: 市场数据对象，用于获取下一个交易日
            
        Returns:
            bool: 是否达到结束时间戳
        """
        try:
            # 使用市场的 get_next_trading_day 方法获取下一个交易日
            next_timestamp = market.get_next_trading_day(self.current_timestamp)
            
            if next_timestamp is None:
                self.logger.info("没有下一个交易日，模拟结束")
                self.current_timestamp = self.end_timestamp
                return True
            
            # 检查是否超过结束时间
            if next_timestamp > self.end_timestamp:
                self.logger.info("下一个交易日已超过结束时间，模拟结束")
                self.current_timestamp = self.end_timestamp
                return True
            
            # 更新当前时间戳
            self.current_timestamp = next_timestamp
            self.logger.debug(f"时间步进完成: {self.current_timestamp}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"时间步进失败: {str(e)}")
            raise