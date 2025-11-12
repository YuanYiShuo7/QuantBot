from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
import json
import os
from pathlib import Path
from ..schemas.account_schema import AccountSchema

class RewardInterface(ABC):
    """奖励系统接口基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化奖励系统
        
        Args:
            config: 配置参数
        """
        pass
    
    @abstractmethod
    def record_trajectory(self, timestamp: datetime, account: AccountSchema, 
                         prompt:str, output: str) -> None:
        """记录Agent的决策轨迹
        
        Args:
            timestamp: 决策时间戳
            account: 账户状态信息
            interaction: 交互信息，包含prompt和output
        """
        pass
    
    @abstractmethod
    def calculate_score(self) -> None:
        """为轨迹计算评分
        """
        pass
    
    @abstractmethod
    def persist_data(self) -> bool:
        """用csv文件持久化轨迹和评分数据
        Returns:
            是否成功保存
        """
        pass