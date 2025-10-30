from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
from decimal import Decimal

class ActionType(str, Enum):
    """动作类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TradeAction(BaseModel):
    """交易动作基类"""
    action_type: ActionType = Field(..., description="动作类型")
    symbol: str = Field(..., description="股票代码", regex=r"^\d{6}$")
    timestamp: datetime = Field(default_factory=datetime.now, description="动作时间戳")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isdigit() or len(v) != 6:
            raise ValueError('股票代码必须是6位数字')
        return v

class BuyAction(TradeAction):
    """买入动作"""
    action_type: ActionType = Field(default=ActionType.BUY, description="买入动作")
    quantity: int = Field(..., gt=0, description="买入数量(股)", example=1000)
    price: Optional[float] = Field(None, gt=0, description="指定价格，None表示市价")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v % 100 != 0:
            raise ValueError('A股交易数量必须是100的整数倍')
        return v

class SellAction(TradeAction):
    """卖出动作"""
    action_type: ActionType = Field(default=ActionType.SELL, description="卖出动作")
    quantity: int = Field(..., gt=0, description="卖出数量(股)")
    price: Optional[float] = Field(None, gt=0, description="指定价格，None表示市价")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v % 100 != 0:
            raise ValueError('A股交易数量必须是100的整数倍')
        return v

class HoldAction(TradeAction):
    """持有动作"""
    action_type: ActionType = Field(default=ActionType.HOLD, description="持有动作")

# 动作联合类型
class ActionSpace(Union[BuyAction, SellAction, HoldAction]):
    """动作空间联合类型"""
    pass