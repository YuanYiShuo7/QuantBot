from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

from model import *
from config import ObservationConfig, MarketConfig

class IDataProvider(ABC):
    """数据提供者接口"""
    
    @abstractmethod
    def get_realtime_market_data(self, indices: list[str], timestamp: datetime) -> list[MarketIndex]:
        """获取实时大盘数据"""
        pass
    
    @abstractmethod
    def get_realtime_stock_data(self, symbols: list[str], timestamp: datetime) -> list[StockTickData]:
        """获取实时股票数据"""
        pass
    
    @abstractmethod
    def get_historical_market_data(self, indices: list[str], granularity: TimeGranularity, 
                                 start_time: datetime, end_time: datetime) -> list[MarketIndex]:
        """获取历史大盘数据"""
        pass
    
    @abstractmethod
    def get_historical_stock_data(self, symbols: list[str], granularity: TimeGranularity,
                                start_time: datetime, end_time: datetime) -> list[StockTickData]:
        """获取历史股票数据"""
        pass

class IAccountProvider(ABC):
    """账户数据提供者接口"""
    
    @abstractmethod
    def get_account_info(self, timestamp: datetime) -> AccountInfo:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_positions(self, timestamp: datetime) -> list[Position]:
        """获取持仓信息"""
        pass