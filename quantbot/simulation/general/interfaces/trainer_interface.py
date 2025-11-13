from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

class TrainerInterface(ABC):
    """训练器接口基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化训练器
        
        Args:
            config: 配置参数，包含：
                - database_path: 轨迹数据CSV文件路径
                - base_model: 基础模型名称
                - lora_config: LoRA配置参数
                - ppo_config: PPO算法参数
                - training_params: 训练超参数
        """
        pass
    
    @abstractmethod
    def train(self, components: Dict[str, Any]) -> bool:
        """执行训练
        
        Args:
            components: 包含所有组件的字典
            
        Returns:
            bool: 训练是否成功
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> bool:
        """保存训练检查点
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> bool:
        """加载训练检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            bool: 是否加载成功
        """
        pass