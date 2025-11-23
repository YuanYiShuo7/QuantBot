from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..schemas.account_schema import AccountSchema 
from ..schemas.order_schema import OrderFormSchema, OrderResultSchema
from ....firm.general.schemas.action_schema import ActionSchemaUnion
from ..schemas.market_schema import MarketSchema
class ExchangeInterface(ABC):
    """模拟交易所接口类"""
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化模拟交易所"""
        pass

    @abstractmethod    
    def update_orders(self, actions: List[ActionSchemaUnion], account, timestamp: datetime):
        pass

    @abstractmethod
    def check_and_execute_orders(self, market, account, timestamp: datetime) -> List[OrderResultSchema]:
        pass