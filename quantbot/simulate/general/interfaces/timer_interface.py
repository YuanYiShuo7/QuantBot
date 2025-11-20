from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime


class TimerInterface(ABC):
    """计时器接口抽象类 - 定义时间管理的标准接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化计时器
        
        Args:
            config: 配置字典，包含计时器相关配置参数，包含
            start_timestamp: 开始时间戳
            current_timestamp: 当前时间戳
            end_timestamp: 结束时间戳
        """
        pass

    @abstractmethod
    def get_current_timestamp(self) -> datetime:
        """获取当前时间戳"""
        pass

    @abstractmethod
    def set_current_timestamp(self, current_timestamp: datetime):
        """设置当前时间戳"""
        pass

    @abstractmethod
    def step(self, market) -> bool:
        """推进时间步长"""
        pass
