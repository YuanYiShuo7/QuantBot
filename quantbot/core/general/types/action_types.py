from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from order_types import OrderInterface
from enum import Enum
class ActionType(str, Enum):
    """操作类型枚举"""
    ADD_ORDER = "ADD_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    NONE = "NONE"
class BaseAction(BaseModel):
    """操作基类"""
    action_type: ActionType = Field(..., description="操作类型")
    timestamp: datetime = Field(default_factory=datetime.now, description="操作时间戳")
    reasoning: str = Field(..., description="操作推理过程说明")

class AddOrderAction(BaseAction):
    """增加订单操作"""
    action_type: ActionType = Field(ActionType.ADD_ORDER, description="操作类型")
    order_interface: OrderInterface = Field(..., description="要添加的订单")

class CancelOrderAction(BaseAction):
    """移除订单操作"""
    action_type: ActionType = Field(ActionType.CANCEL_ORDER, description="操作类型")
    order_id: str = Field(..., description="要移除的订单ID")

class NoneAction(BaseAction):
    """无操作"""
    action_type: ActionType = Field(ActionType.NONE, description="操作类型")


# 联合类型，包含所有可能的操作
ActionSpace = Union[AddOrderAction, CancelOrderAction, NoneAction]