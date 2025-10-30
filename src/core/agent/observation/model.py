from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, Field
from enum import Enum

class TimeGranularity(str, Enum):
    MINUTE = "minute"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class MarketIndex(BaseModel):
    """大盘指数基础信息"""
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

class StockTickData(BaseModel):
    """股票实时tick数据"""
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

class Position(BaseModel):
    """持仓信息"""
    symbol: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    quantity: int = Field(..., description="持仓数量")
    available_quantity: int = Field(..., description="可用数量")
    cost_price: float = Field(..., description="成本价")
    current_price: float = Field(..., description="当前价")
    market_value: float = Field(..., description="市值")
    profit_loss: float = Field(..., description="浮动盈亏")
    profit_loss_rate: float = Field(..., description="盈亏比例")
    timestamp: datetime = Field(..., description="时间戳")

class AccountInfo(BaseModel):
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

class HistoricalKLine(BaseModel):
    """历史K线数据"""
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

class ObservationData(BaseModel):
    """完整的观测数据"""
    timestamp: datetime = Field(..., description="观测时间点")
    
    # 实时信息
    realtime_market: List[MarketIndex] = Field(..., description="实时大盘信息")
    realtime_stocks: List[StockTickData] = Field(..., description="实时自选股票信息")
    realtime_account: AccountInfo = Field(..., description="实时账户信息")
    
    # 历史信息
    historical_market: Dict[TimeGranularity, List[MarketIndex]] = Field(..., description="历史大盘信息")
    historical_stocks: Dict[TimeGranularity, List[StockTickData]] = Field(..., description="历史自选股票信息")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")