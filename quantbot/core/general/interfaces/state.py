from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..types.obervation_types import ObervationSpace

class StateInterface(ABC):
    """交易数据接口抽象类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化接口
        Args:
            config: 配置参数字典
        """
        pass
    
    @abstractmethod
    def initialize_observation(self) -> ObervationSpace:
        """初始化交易状态
        Returns:
            TradingState: 初始化的交易状态对象
        """
        pass
    
    @abstractmethod
    def update_obervation(self, state: ObervationSpace) -> ObervationSpace:
        """更新交易状态
        Args:
            state: 当前交易状态
        Returns:
            TradingState: 更新后的交易状态
        """
        pass
    
    @abstractmethod
    def get_obervation(self) -> ObervationSpace:
        """获取当前交易状态
        Returns:
            TradingState: 当前交易状态
        """
        pass