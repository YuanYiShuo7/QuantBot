from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class OrderStatus(str, Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"

class OrderType(str, Enum):
    """订单类型枚举"""
    BUY = "BUY"
    SELL = "SELL"

class OrderInterface(BaseModel):
    """用户界面订单接口类 - 简化的订单对象，专为模型输入和UI设计"""
    
    symbol: str = Field(..., description="股票代码", example="000001")
    order_type: OrderType = Field(..., description="订单类型: BUY/SELL")
    price: float = Field(..., description="价格", example=12.50)
    quantity: float = Field(..., description="数量", example=100)

    @validator('symbol')
    def symbol_must_be_valid(cls, v):
        """验证股票代码格式"""
        if not v or len(v) != 6:
            raise ValueError('股票代码必须为6位数字')
        if not v.isdigit():
            raise ValueError('股票代码必须为数字')
        return v
    
    @validator('price')
    def price_must_be_positive(cls, v):
        """验证价格必须大于0"""
        if v <= 0:
            raise ValueError('价格必须大于0')
        # A股价格精度为2位小数
        return round(v, 2)
    
    @validator('quantity')
    def quantity_must_be_positive_multiple(cls, v):
        """验证数量必须为正数且为100的整数倍（A股规则）"""
        if v <= 0:
            raise ValueError('数量必须大于0')
        # A股最小交易单位为100股
        if v % 100 != 0:
            raise ValueError('数量必须是100的整数倍')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "000001",
                "order_type": "BUY",
                "price": 12.50,
                "quantity": 100
            }
        }

class Order(BaseModel):
    """订单基类"""
    order_id: str = Field(..., description="订单唯一标识")
    symbol: str = Field(..., description="交易对")
    order_type: OrderType = Field(..., description="订单类型: buy/sell")
    price: float = Field(..., description="价格")
    quantity: float = Field(..., description="数量")
    timestamp: datetime = Field(default_factory=datetime.now, description="订单创建时间")
    status: OrderStatus = Field(OrderStatus.PENDING, description="订单状态")
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('价格必须大于0')
        return v
    
    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('数量必须大于0')
        return v
    
class OrderList(BaseModel):
    """订单列表"""
    orders: List[Order] = Field(default_factory=list, description="订单列表")
    
    def add_order(self, order: Order):
        """添加订单到列表"""
        self.orders.append(order)   