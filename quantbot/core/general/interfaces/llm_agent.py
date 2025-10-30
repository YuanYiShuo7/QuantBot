from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from ..types.action_types import ActionSpace

class LLMAgentInterface(ABC):
    """LLM Agent 接口抽象类"""
    
    @abstractmethod
    def __init__(self, agent_config: Dict[str, Any] = None):
        """初始化LLM Agent
        Args:
            agent_config: Agent配置参数，包含模型配置、策略参数等
        """
        pass
    
    @abstractmethod
    def get_state_observation(self) -> Dict[str, Any]:
        """获取Agent观察到的状态信息
        用于构建LLM的prompt输入
        
        Returns:
            Dict[str, Any]: 状态观察字典，包含市场数据、持仓、账户等信息
        """
        pass
    
    @abstractmethod
    def generate_action(self, state_observation: Dict[str, Any]) -> Dict[str, Any]:
        """基于状态观察生成动作
        Args:
            state_observation: 状态观察信息
        Returns:
            Dict[str, Any]: 动作决策，包含动作类型、标的、数量等
        """
        pass
    
    @abstractmethod
    def update_learning(self, reward: float, next_state: Dict[str, Any], done: bool = False):
        """根据奖励信号更新Agent学习状态
        Args:
            reward: 奖励值
            next_state: 下一状态观察
            done: 是否结束episode
        """
        pass