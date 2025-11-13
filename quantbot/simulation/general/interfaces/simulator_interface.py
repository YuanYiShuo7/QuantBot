from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class SimulatorInterface(ABC):
    """模拟器接口基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化模拟器
        
        Args:
            config: 配置参数，包含JSON配置文件路径等
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """运行模拟器"""
        pass
