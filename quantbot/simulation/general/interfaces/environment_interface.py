from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

class EnvironmentInterface(ABC):
    """交易环境接口"""
    @abstractmethod    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化交易环境
        Args:
            config: 环境配置参数，包含市场、账户、交易所,交易模式，起始时间，终止时间等配置
        """
        pass
    
    @abstractmethod    
    def step(self) -> List[Dict[str, Any]] | None:
        """执行单个交易时间步
        Returns:
            List[Dict[str, Any]] | None: 每个时间步的交互轨迹列表，包含:
                - prompt: 输入prompt
                - output: 模型输出
                - rewards: 奖励值列表
        """
        pass