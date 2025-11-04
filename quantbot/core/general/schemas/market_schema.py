from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

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

class MarketSchema(BaseModel):
    """市场相关数据 - 外部市场环境，不可控部分"""
    timestamp: datetime = Field(..., description="市场数据时间戳")
    
    # 市场状态
    market_status: str = Field(..., description="市场状态")
    
    # 市场数据
    market_indices: Dict[str, IndexData] = Field(default_factory=dict, description="大盘指数数据，key为指数代码")
    stock_data: Dict[str, StockData] = Field(default_factory=dict, description="股票数据，key为股票代码")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T09:30:00",
                "market_status": "open",
                "market_indices": {
                    "000001": {
                        "real_time": {
                            "symbol": "000001",
                            "name": "上证指数",
                            "timestamp": "2024-01-15T09:30:00",
                            "current_price": 3200.0,
                            "change": 15.0,
                            "change_rate": 0.47,
                            "volume": 150000000,
                            "turnover": 18000000000.0,
                            "open": 3190.0,
                            "high": 3210.0,
                            "low": 3185.0,
                            "pre_close": 3185.0,
                            "pe_ratio": 15.0,
                            "pb_ratio": 1.5,
                            "amplitude": 0.78,
                            "turnover_rate": None
                        },
                        "daily_klines": [...],
                        "weekly_klines": [...],
                        "monthly_klines": [...]
                    }
                },
                "stock_data": {
                    "000001": {
                        "real_time": {
                            "symbol": "000001",
                            "name": "平安银行",
                            "timestamp": "2024-01-15T09:30:00",
                            "current_price": 13.0,
                            "open": 12.8,
                            "high": 13.2,
                            "low": 12.7,
                            "pre_close": 12.8,
                            "change": 0.2,
                            "change_rate": 1.56,
                            "volume": 1000000,
                            "turnover": 13000000.0,
                            "bid_prices": [12.99, 12.98, 12.97, 12.96, 12.95],
                            "bid_volumes": [50000, 60000, 70000, 80000, 90000],
                            "ask_prices": [13.01, 13.02, 13.03, 13.04, 13.05],
                            "ask_volumes": [40000, 50000, 60000, 70000, 80000],
                            "pe_ratio": 8.5,
                            "pb_ratio": 0.9,
                            "amplitude": 3.91,
                            "turnover_rate": 0.5,
                            "volume_ratio": 1.2,
                            "committee": 0.1,
                            "main_net_inflow": 5000000.0,
                            "large_net_inflow": 3000000.0,
                            "medium_net_inflow": 1000000.0,
                            "small_net_inflow": 1000000.0
                        },
                        "daily_klines": [...],
                        "weekly_klines": [...],
                        "monthly_klines": [...]
                    }
                }
            }
        }