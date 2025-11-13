from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import pickle
from pathlib import Path

from ..schemas.market_schema import MarketSchema

class MarketInterface(ABC):
    """市场数据接口抽象类 - 定义市场数据获取和管理的标准接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化市场接口
        
        Args:
            config: 配置字典，包含以下可选参数：
                - cache_dir: 缓存目录路径
                - default_symbols: 默认关注的股票代码列表
                - default_indices: 默认关注的指数代码列表
                - start_timestamp: 模拟单个交易轮次起始时间
                - end_timestamp:  模拟单个交易轮次终止时间
                - step_interval: 市场实时数据更新时间步长间隔
        """
        pass

    @abstractmethod
    def get_market_schema(self) -> MarketSchema:
        """获取当前市场数据的 MarketSchema 对象"""
        pass

    @abstractmethod 
    def set_market_schema(self, market_schema: MarketSchema):
        """设置当前市场数据的 MarketSchema 对象"""
        pass
    
    @abstractmethod
    def initialize_market_data_cache(self) -> bool:
        pass

    @abstractmethod
    def update_market_from_data_cache(self, timestamp: datetime) -> MarketSchema:
        """从本地缓存加载市场数据并更新 MarketSchema"""
        pass
    
    @abstractmethod
    def format_market_info_for_prompt(self) -> str:
        """将市场信息 MarketSchema 转换为 Agent 的 Prompt 文本"""
        pass