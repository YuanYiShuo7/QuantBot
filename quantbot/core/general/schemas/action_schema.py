from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from ..schemas.order_schema import OrderFormSchema
from enum import Enum

class ActionType(str, Enum):
    """操作类型枚举"""
    ADD_ORDER = "ADD_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    NONE = "NONE"

class BaseActionSchema(BaseModel):
    """操作基类数据模型"""
    reasoning: str = Field(..., description="操作推理过程说明")
    action_type: ActionType = Field(..., description="操作类型")

class AddOrderActionSchema(BaseActionSchema):
    """增加订单操作数据模型"""
    action_type: ActionType = Field(ActionType.ADD_ORDER, description="操作类型")
    order_form: OrderFormSchema = Field(..., description="要添加的订单表单数据")

class CancelOrderActionSchema(BaseActionSchema):
    """取消订单操作数据模型"""
    action_type: ActionType = Field(ActionType.CANCEL_ORDER, description="操作类型")
    order_id: str = Field(..., description="要取消的订单ID")

class NoneActionSchema(BaseActionSchema):
    """无操作数据模型"""
    action_type: ActionType = Field(ActionType.NONE, description="操作类型")

ActionSchemaUnion = Union[AddOrderActionSchema, CancelOrderActionSchema, NoneActionSchema]