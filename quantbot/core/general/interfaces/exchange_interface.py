from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from ..schemas.account_schema import AccountSchema 
from ..schemas.order_schema import OrderFormSchema, OrderResultSchema
from ..schemas.action_schema import ActionSchemaUnion

class ExchangeInterface(ABC):
    """模拟交易所接口类"""
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """初始化模拟交易所"""
        pass

    @abstractmethod    
    def update_orders(self, action: List[ActionSchemaUnion], account, timestamp: datetime):
        pass

    @abstractmethod
    def check_and_execute_orders(self, market, account, timestamp: datetime) -> List[OrderResultSchema]:
        pass