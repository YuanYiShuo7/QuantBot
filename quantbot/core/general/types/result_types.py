from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
from order_types import OrderStatus
from order_types import Order

class Result(BaseModel):
    """交易结果基类"""
    order:Order = Field(..., description="相关订单信息")
    status: OrderStatus = Field(..., description="订单状态详情")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    
    # 成功时的成交信息（可选）
    executed_price: Optional[float] = Field(None, description="实际成交价格")
    executed_quantity: Optional[float] = Field(None, description="实际成交量")
    remaining_quantity: Optional[float] = Field(None, description="剩余未成交量")
    fee: Optional[float] = Field(None, description="手续费")
    
    # 失败时的错误信息（可选）
    error_reason: Optional[str] = Field(None, description="失败原因")
    
    @classmethod
    def success(cls, 
                content: str,
                order_id: str,
                executed_price: float,
                executed_quantity: float,
                remaining_quantity: float = 0,
                fee: float = 0,
                status: OrderStatus = OrderStatus.SUCCESS):
        """创建成功消息"""
        return cls(
            order=Order(order_id=order_id),
            status=status,
            content=content,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            remaining_quantity=remaining_quantity,
            fee=fee
        )
    
    @classmethod
    def failure(cls, 
                content: str,
                error_reason: str,
                status: OrderStatus = OrderStatus.FAILED,
                order_id: Optional[str] = None):
        """创建失败消息"""
        return cls(
            order=Order(order_id=order_id),
            status=status,
            content=content,
            error_reason=error_reason,
        )
    
class ResultList(BaseModel):
    """交易结果列表"""
    results: List[Result] = Field(default_factory=list, description="交易结果列表")
    
    def add_result(self, result: Result):
        """添加交易结果"""
        self.results.append(result)