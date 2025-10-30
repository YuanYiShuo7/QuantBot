from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..types.action_types import ActionSpace
from ..types.obervation_types import ObervationSpace

class EnvironmentInterface(ABC):
    """交易环境接口"""
    
    @abstractmethod
    def reset(self, initial_capital: float = 100000.0) -> StateObervationSpace:
        """
        重置环境状态
        
        Args:
            initial_capital: 初始资金
            
        Returns:
            TradingState: 初始状态
        """
        pass
    
    @abstractmethod
    def step(self, action: ActionSpace) -> Tuple[StateObervationSpace, float, bool, Dict[str, Any]]:
        """
        执行动作并推进环境
        
        Args:
            action: 交易动作
            
        Returns:
            Tuple[TradingState, float, bool, Dict[str, Any]]: 
                - 新状态
                - 奖励值
                - 是否结束
                - 附加信息
        """
        pass
    
    @abstractmethod
    def get_state(self) -> StateObervationSpace:
        """获取当前环境状态"""
        pass
    
    @abstractmethod
    def get_actions(self, state: StateObervationSpace) -> List[ActionSpace]:
        """获取在当前状态下有效的动作列表"""
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Any:
        """渲染环境状态"""
        pass