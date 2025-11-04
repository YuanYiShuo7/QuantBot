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

class OrderFormSchema(BaseModel):
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

class OrderSchema(BaseModel):
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
    
class OrderResultSchema(BaseModel):
    """订单结果数据模型 - 仅当订单成功或失败时返回相关元数据"""
    
    order_id: str = Field(..., description="订单唯一标识")
    symbol: str = Field(..., description="股票代码")
    order_type: OrderType = Field(..., description="订单类型")
    status: OrderStatus = Field(..., description="订单最终状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="结果时间戳")
    
    # 成功时的成交信息
    executed_quantity: Optional[float] = Field(None, description="实际成交量")
    executed_price: Optional[float] = Field(None, description="实际成交均价")
    executed_amount: Optional[float] = Field(None, description="实际成交金额")
    commission: Optional[float] = Field(None, description="手续费")
    
    # 失败时的错误信息
    error_message: Optional[str] = Field(None, description="失败错误信息")
    
    # 验证逻辑
    @validator('executed_quantity')
    def validate_executed_quantity(cls, v, values):
        """验证成交量"""
        if v is not None and v <= 0:
            raise ValueError('成交量必须大于0')
        return v
    
    @validator('executed_price')
    def validate_executed_price(cls, v, values):
        """验证成交价格"""
        if v is not None and v <= 0:
            raise ValueError('成交价格必须大于0')
        return v
    
    @validator('executed_amount')
    def validate_executed_amount(cls, v, values):
        """验证成交金额"""
        if v is not None and v < 0:
            raise ValueError('成交金额不能为负数')
        return v

    class Config:
        schema_extra = {
            "examples": {
                "success": {
                    "order_id": "ORD_123456",
                    "symbol": "000001",
                    "order_type": "BUY",
                    "status": "success",
                    "timestamp": "2024-01-15T09:30:00",
                    "executed_quantity": 100,
                    "executed_price": 12.45,
                    "executed_amount": 1245.0,
                    "commission": 3.74,
                    "stamp_duty": 0.0,  # 买入免印花税
                    "transfer_fee": 0.1,
                    "total_fee": 3.84,
                    "net_amount": 1248.84,
                    "error_message": None,
                    "error_code": None
                },
                "failed": {
                    "order_id": "ORD_123457",
                    "symbol": "000002",
                    "order_type": "SELL",
                    "status": "failed",
                    "timestamp": "2024-01-15T09:31:00",
                    "executed_quantity": None,
                    "executed_price": None,
                    "executed_amount": None,
                    "commission": None,
                    "stamp_duty": None,
                    "transfer_fee": None,
                    "total_fee": None,
                    "net_amount": None,
                    "error_message": "资金不足",
                    "error_code": "INSUFFICIENT_FUNDS"
                }
            }
        }