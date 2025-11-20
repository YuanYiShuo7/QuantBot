from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional

class EnvironmentInterface(ABC):
    """交易环境接口"""
    @abstractmethod    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化交易环境
        Args:
            config: 环境配置参数，包含起始时间，终止时间等配置
        """
        pass
    
    @abstractmethod    
    def step(self, timer, account, exchange, llm_agent, market, reward) -> bool:
        """执行单个交易时间步
        Returns:
            bool: 是否达到终止条件
        """
        pass