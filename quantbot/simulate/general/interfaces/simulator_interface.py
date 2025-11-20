from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class SimulatorInterface(ABC):
    """模拟器接口基类"""
    
    @abstractmethod
    def __init__(self, components_config):
        """初始化模拟器
        
        Args:
            components_config: 各个组件的配置参数
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """运行模拟器"""
        pass
