from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from order_types import OrderList
class TimeGranularity(str, Enum):
    MINUTE = "minute"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class KLineData(BaseModel):
    """K线数据基础类"""
    symbol: str = Field(..., description="标的代码")
    name: str = Field(..., description="标的名称")
    granularity: TimeGranularity = Field(..., description="时间粒度")
    timestamp: datetime = Field(..., description="时间戳")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    close: float = Field(..., description="收盘价")
    volume: int = Field(..., description="成交量")
    turnover: float = Field(..., description="成交额")
    change: float = Field(..., description="涨跌额")
    change_rate: float = Field(..., description="涨跌幅")
    amplitude: float = Field(..., description="振幅")
    turnover_rate: Optional[float] = Field(None, description="换手率")

class IndexRealTimeData(BaseModel):
    """指数实时数据"""
    symbol: str = Field(..., description="指数代码")
    name: str = Field(..., description="指数名称")
    timestamp: datetime = Field(..., description="时间戳")
    current_price: float = Field(..., description="当前价格")
    change: float = Field(..., description="涨跌额")
    change_rate: float = Field(..., description="涨跌幅")
    volume: float = Field(..., description="成交量")
    turnover: float = Field(..., description="成交额")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    pre_close: float = Field(..., description="前收盘价")
    
    # 技术指标
    pe_ratio: Optional[float] = Field(None, description="市盈率")
    pb_ratio: Optional[float] = Field(None, description="市净率")
    amplitude: float = Field(..., description="振幅")
    turnover_rate: Optional[float] = Field(None, description="换手率")

class IndexData(BaseModel):
    """指数完整数据"""
    real_time: IndexRealTimeData = Field(..., description="实时数据")
    daily_klines: List[KLineData] = Field(default_factory=list, description="日K线数据")
    weekly_klines: List[KLineData] = Field(default_factory=list, description="周K线数据")
    monthly_klines: List[KLineData] = Field(default_factory=list, description="月K线数据")

class StockRealTimeData(BaseModel):
    """股票实时数据"""
    symbol: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    timestamp: datetime = Field(..., description="时间戳")
    
    # 价格信息
    current_price: float = Field(..., description="当前价")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    pre_close: float = Field(..., description="前收盘价")
    
    # 涨跌信息
    change: float = Field(..., description="涨跌额")
    change_rate: float = Field(..., description="涨跌幅")
    
    # 成交信息
    volume: int = Field(..., description="成交量")
    turnover: float = Field(..., description="成交额")
    
    # 盘口信息
    bid_prices: List[float] = Field(..., description="买一至买五价格")
    bid_volumes: List[int] = Field(..., description="买一至买五量")
    ask_prices: List[float] = Field(..., description="卖一至卖五价格")
    ask_volumes: List[int] = Field(..., description="卖一至卖五量")
    
    # 技术指标
    pe_ratio: Optional[float] = Field(None, description="市盈率")
    pb_ratio: Optional[float] = Field(None, description="市净率")
    amplitude: float = Field(..., description="振幅")
    turnover_rate: float = Field(..., description="换手率")
    volume_ratio: Optional[float] = Field(None, description="量比")
    committee: Optional[float] = Field(None, description="委比")
    
    # 资金流向
    main_net_inflow: Optional[float] = Field(None, description="主力净流入")
    large_net_inflow: Optional[float] = Field(None, description="大单净流入")
    medium_net_inflow: Optional[float] = Field(None, description="中单净流入")
    small_net_inflow: Optional[float] = Field(None, description="小单净流入")

class StockData(BaseModel):
    """股票完整数据"""
    real_time: StockRealTimeData = Field(..., description="实时数据")
    daily_klines: List[KLineData] = Field(default_factory=list, description="日K线数据")
    weekly_klines: List[KLineData] = Field(default_factory=list, description="周K线数据")
    monthly_klines: List[KLineData] = Field(default_factory=list, description="月K线数据")

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

class ObervationSpace(BaseModel):
    """交易状态观测总入口类"""
    timestamp: datetime = Field(..., description="状态时间戳")
    
    # 市场数据
    market_indices: Dict[str, IndexData] = Field(default_factory=dict, description="大盘指数数据，key为指数代码")
    stock_data: Dict[str, StockData] = Field(default_factory=dict, description="股票数据，key为股票代码")
    
    # 投资组合数据
    account_info: AccountData = Field(..., description="账户信息")
    positions: List[PositionData] = Field(default_factory=dict, description="持仓信息，key为股票代码")

    # 订单信息
    orders: OrderList = Field(default_factory=OrderList, description="当前订单列表")
    
    # 配置信息
    watch_list: List[str] = Field(default_factory=list, description="自选股列表")
    market_index_list: List[str] = Field(default_factory=list, description="关注的大盘指数列表")
    
    # 环境状态
    trading_enabled: bool = Field(..., description="是否可交易")
    market_status: str = Field(..., description="市场状态")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T09:30:00",
                "market_indices": {
                    "000001": {
                        "real_time": {...},
                        "daily_klines": [...],
                        "weekly_klines": [...],
                        "monthly_klines": [...]
                    }
                },
                "stock_data": {
                    "000001": {
                        "real_time": {...},
                        "daily_klines": [...],
                        "weekly_klines": [...],
                        "monthly_klines": [...]
                    }
                },
                "account_info": {...},
                "positions": {...},
                "orders": {...},
                "watch_list": ["000001", "000002"],
                "market_index_list": ["000001", "000300"],
                "trading_enabled": True,
                "market_status": "open"
            }
        }