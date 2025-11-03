from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..types.obervation_types import ObervationSpace
from ..types.result_types import ResultList
from ..types.order_types import OrderList
class ObservationInterface(ABC):
    """交易数据接口抽象类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化接口
        Args:
            config: 配置参数字典
        """
        pass
    
    @abstractmethod
    def initialize_observation(self, timestamp, market_data) -> ObervationSpace:
        """初始化交易状态
        Returns:
            ObervationSpace: 初始化的交易状态对象
        """
        pass
    
    @abstractmethod
    def update_obervation(self, timestamp, market_data, order_results: ResultList) -> ObervationSpace:
        """更新交易状态
        Args:
            market_data: 最新市场数据
            orders: 当前订单列表
            order_results: 订单执行结果列表
        Returns:
            ObervationSpace: 更新后的交易状态
        """
        pass
    
    @abstractmethod
    def get_obervation(self) -> ObervationSpace:
        """获取当前交易状态
        Returns:
            ObervationSpace: 当前交易状态
        """
        pass