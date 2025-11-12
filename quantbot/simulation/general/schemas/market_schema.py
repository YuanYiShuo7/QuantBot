from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class TimeGranularity(str, Enum):
    MINUTE = "MINUTE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

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
    volume: float = Field(..., description="成交量")
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
    
    # 盘口信息
    bid_prices: List[float] = Field(..., description="买一至买五价格")
    bid_volumes: List[float] = Field(..., description="买一至买五量")
    ask_prices: List[float] = Field(..., description="卖一至卖五价格")
    ask_volumes: List[float] = Field(..., description="卖一至卖五量")

    volume: float = Field(..., description="成交量")
    turnover: float = Field(..., description="成交额")

    # 涨跌信息
    change: float = Field(..., description="涨跌额")
    change_rate: float = Field(..., description="涨跌幅")

class StockData(BaseModel):
    """股票完整数据"""
    real_time: StockRealTimeData = Field(..., description="实时数据")
    daily_klines: List[KLineData] = Field(default_factory=list, description="日K线数据")
    weekly_klines: List[KLineData] = Field(default_factory=list, description="周K线数据")
    monthly_klines: List[KLineData] = Field(default_factory=list, description="月K线数据")

class MarketSchema(BaseModel):
    """市场相关数据 - 外部市场环境，不可控部分"""
    timestamp: datetime = Field(..., description="市场数据时间戳")
    
    # 市场状态
    market_status: str = Field(..., description="市场状态")
    
    # 市场数据
    market_indices: Dict[str, IndexData] = Field(default_factory=dict, description="大盘指数数据，key为指数代码")
    stock_data: Dict[str, StockData] = Field(default_factory=dict, description="股票数据，key为股票代码")