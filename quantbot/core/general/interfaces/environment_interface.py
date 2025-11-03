from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..types.action_types import ActionSpace
from ..types.obervation_types import ObervationSpace

class EnvironmentInterface(ABC):
    """交易环境接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        pass
    
    @abstractmethod
    def step(self, action: ActionSpace) -> Tuple[ObervationSpace, float, bool, Dict[str, Any]]:
        """
        执行动作并推进环境
        
        Args:
            action: 交易动作
            
        Returns:
            Tuple[ObervationSpace, float, bool, Dict[str, Any]]: 
                - 新观测值
                - 奖励值
                - 是否结束
                - 附加信息
        """
        pass