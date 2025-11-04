from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from ..schemas.action_schema import ActionSchemaUnion


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
    def generate_prompt(self, account, market) -> str:
        """生成完整的prompt文本
        Args:
            account: 账户信息
            market: 市场信息
        Returns:
            str: 完整的prompt文本
        """
        pass
    
    @abstractmethod
    def generate_output(self, prompt: str) -> str:
        """基于prompt生成输出和掩码
        Args:
            prompt: 输入prompt文本
        Returns:
            str: 模型输出文本
        """
        pass
    
    @abstractmethod
    def parse_action(self, output: str) -> List[ActionSchemaUnion]:
        """解析输出文本为动作
        Args:
            output: 模型输出文本
        Returns:
            ActionSchemaUnion: 解析后的动作对象
        Raises:
            ActionParseError: 当解析失败时
        """
        pass
    
    @abstractmethod
    def update_learning(self, traces: List[Dict[str, Any]]) -> None:
        """根据交互轨迹更新Agent学习状态
        Args:
            traces: 交互轨迹列表，每个元素包含:
                - prompt: 输入prompt
                - output: 模型输出
                - rewards: 奖励值列表
        """
        pass