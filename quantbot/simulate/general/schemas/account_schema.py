from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from ..schemas.order_schema import OrderSchema

class PositionData(BaseModel):
    """持仓信息"""
    timestamp: datetime = Field(..., description="时间戳")
    symbol: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    quantity: int = Field(..., description="持仓数量")
    available_quantity: int = Field(..., description="可用数量")
    cost_price: float = Field(..., description="成本价")
    current_price: float = Field(..., description="当前价")
    market_value: float = Field(..., description="市值")
    profit_loss: float = Field(..., description="浮动盈亏")
    profit_loss_rate: float = Field(..., description="盈亏比例")

class AccountData(BaseModel):
    """账户信息"""
    timestamp: datetime = Field(..., description="时间戳")
    total_assets: float = Field(..., description="总资产")
    net_assets: float = Field(..., description="净资产")
    available_cash: float = Field(..., description="可用资金")
    market_value: float = Field(..., description="持仓市值")
    total_profit_loss: float = Field(..., description="总浮动盈亏")
    total_profit_loss_rate: float = Field(..., description="总盈亏比例")
    position_rate: float = Field(..., description="持仓比例")


class AccountSchema(BaseModel):
    """账户相关数据 - 所有能由人把控的部分"""
    timestamp: datetime = Field(..., description="账户状态时间戳")
    
    # 账户核心信息
    account_info: AccountData = Field(..., description="账户信息")
    positions: List[PositionData] = Field(default_factory=dict, description="持仓信息，key为股票代码")
    
    # 交易活动
    orders: List[OrderSchema] = Field(..., description="当前订单列表")
    
    # 用户配置
    watch_list: List[str] = Field(default_factory=list, description="自选股列表")
    market_index_list: List[str] = Field(default_factory=list, description="关注的大盘指数列表")