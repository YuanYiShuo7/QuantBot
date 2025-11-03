from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from ..types.action_types import ActionSpace
from ..types.obervation_types import ObervationSpace


class LLMAgentInterface(ABC):
    """LLM Agent 接口抽象类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化LLM Agent
        Args:
            config: Agent配置参数，包含模型配置、策略参数等
        """
        pass
    
    @abstractmethod
    def generate_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """基于状态观察生成动作
        Args:
            state_observation: 状态观察信息
        Returns:
            Dict[str, Any]: 动作决策，包含动作类型、标的、数量等
        """
        pass
    
    @abstractmethod
    def update_learning(self, rewards):
        """根据奖励信号更新Agent学习状态
        Args:
            rewards: 奖励值列表
            next_state: 下一状态观察
            done: 是否结束episode
        """
        pass