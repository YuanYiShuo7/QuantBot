from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from quantbot.core.general.schemas.order_schema import OrderSchema

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
    today_profit_loss: float = Field(..., description="当日浮动盈亏")
    
    # 风险指标
    position_rate: float = Field(..., description="持仓比例")
    margin_ratio: Optional[float] = Field(None, description="保证金比例")

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
    
    # 交易权限
    trading_enabled: bool = Field(..., description="是否可交易")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T09:30:00",
                "account_info": {
                    "timestamp": "2024-01-15T09:30:00",
                    "total_assets": 1000000.0,
                    "net_assets": 980000.0,
                    "available_cash": 200000.0,
                    "market_value": 800000.0,
                    "total_profit_loss": 50000.0,
                    "total_profit_loss_rate": 5.0,
                    "today_profit_loss": 3000.0,
                    "position_rate": 80.0,
                    "margin_ratio": None
                },
                "positions": {
                    "000001": {
                        "timestamp": "2024-01-15T09:30:00",
                        "symbol": "000001",
                        "name": "平安银行",
                        "quantity": 10000,
                        "available_quantity": 10000,
                        "cost_price": 12.5,
                        "current_price": 13.0,
                        "market_value": 130000.0,
                        "profit_loss": 5000.0,
                        "profit_loss_rate": 4.0
                    }
                },
                "orders": {...},
                "watch_list": ["000001", "000002"],
                "market_index_list": ["000001", "000300"],
                "trading_enabled": True
            }
        }