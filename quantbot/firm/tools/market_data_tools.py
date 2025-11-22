from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libs.agno.tools import Toolkit
from libs.agno.run import RunContext
from pydantic import BaseModel, Field

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


class MarketDataTools(Toolkit):
    """市场数据工具类 - 获取实时股票和指数信息"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="market_data_tools",
            tools=[
                self.get_stock_realtime,
                self.get_index_realtime,
                self.get_multiple_stocks,
                self.get_multiple_indices,
                self.get_stock_basic_info
            ],
            **kwargs
        )
        
        # 股票代码映射表（示例数据）
        self.stock_name_map = {
            "000001": "平安银行",
            "000002": "万科A", 
            "000858": "五粮液",
            "600519": "贵州茅台",
            "601318": "中国平安",
            "600036": "招商银行",
            "000063": "中兴通讯",
            "300059": "东方财富",
            "002415": "海康威视",
            "000333": "美的集团"
        }
        
        # 指数代码映射表
        self.index_name_map = {
            "000001": "上证指数",
            "399001": "深证成指", 
            "399006": "创业板指",
            "000300": "沪深300",
            "000905": "中证500",
            "399005": "中小板指",
            "000016": "上证50",
            "399673": "创业板50"
        }

    def _get_stock_data_from_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        从API获取股票实时数据（模拟实现）
        实际使用时可以替换为真实的股票数据API
        """
        try:
            # 这里使用模拟数据，实际可以接入akshare、tushare、新浪财经等API
            import random
            import time
            
            # 模拟API延迟
            time.sleep(0.1)
            
            # 基础价格（模拟）
            base_price = random.uniform(10, 100)
            change_rate = random.uniform(-0.05, 0.05)
            change = base_price * change_rate
            current_price = base_price + change
            
            return {
                "symbol": symbol,
                "name": self.stock_name_map.get(symbol, f"股票{symbol}"),
                "timestamp": datetime.now(),
                "current_price": round(current_price, 2),
                "open": round(base_price * random.uniform(0.98, 1.02), 2),
                "high": round(current_price * random.uniform(1.0, 1.03), 2),
                "low": round(current_price * random.uniform(0.97, 1.0), 2),
                "bid_prices": [round(current_price * random.uniform(0.995, 0.999), 2) for _ in range(5)],
                "bid_volumes": [random.randint(100, 1000) for _ in range(5)],
                "ask_prices": [round(current_price * random.uniform(1.001, 1.005), 2) for _ in range(5)],
                "ask_volumes": [random.randint(100, 1000) for _ in range(5)],
                "volume": random.randint(1000000, 50000000),
                "turnover": random.uniform(10000000, 500000000),
                "change": round(change, 2),
                "change_rate": round(change_rate * 100, 2)
            }
            
        except Exception as e:
            print(f"获取股票数据时发生错误: {e}")
            return None

    def _get_index_data_from_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        从API获取指数实时数据（模拟实现）
        """
        try:
            import random
            import time
            
            time.sleep(0.1)
            
            # 不同指数的基础价格
            base_prices = {
                "000001": 3000,  # 上证指数
                "399001": 10000, # 深证成指
                "399006": 2000,  # 创业板指
                "000300": 3800,  # 沪深300
                "000905": 6000,  # 中证500
            }
            
            base_price = base_prices.get(symbol, 3000)
            change_rate = random.uniform(-0.03, 0.03)
            change = base_price * change_rate
            current_price = base_price + change
            
            return {
                "symbol": symbol,
                "name": self.index_name_map.get(symbol, f"指数{symbol}"),
                "timestamp": datetime.now(),
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_rate": round(change_rate * 100, 2),
                "volume": random.randint(100000000, 5000000000),
                "turnover": random.uniform(1000000000, 50000000000),
                "open": round(base_price * random.uniform(0.99, 1.01), 2),
                "high": round(current_price * random.uniform(1.0, 1.02), 2),
                "low": round(current_price * random.uniform(0.98, 1.0), 2)
            }
            
        except Exception as e:
            print(f"获取指数数据时发生错误: {e}")
            return None

    def get_stock_realtime(self, symbol: str) -> str:
        """
        获取单只股票的实时数据
        
        Args:
            symbol: 股票代码，6位数字
        """
        try:
            # 验证股票代码
            if not symbol or len(symbol) != 6 or not symbol.isdigit():
                return json.dumps({"error": "股票代码必须为6位数字"}, ensure_ascii=False)
            
            # 获取数据
            stock_data = self._get_stock_data_from_api(symbol)
            if not stock_data:
                return json.dumps({"error": f"获取股票{symbol}数据失败"}, ensure_ascii=False)
            
            # 转换为Pydantic模型并返回JSON
            stock_realtime = StockRealTimeData(**stock_data)
            return json.dumps(stock_realtime.dict(), ensure_ascii=False, default=str)
            
        except Exception as e:
            error_msg = {"error": f"获取股票实时数据时发生错误: {str(e)}"}
            return json.dumps(error_msg, ensure_ascii=False)

    def get_index_realtime(self, symbol: str) -> str:
        """
        获取单个指数的实时数据
        
        Args:
            symbol: 指数代码
        """
        try:
            if not symbol:
                return json.dumps({"error": "指数代码不能为空"}, ensure_ascii=False)
            
            # 获取数据
            index_data = self._get_index_data_from_api(symbol)
            if not index_data:
                return json.dumps({"error": f"获取指数{symbol}数据失败"}, ensure_ascii=False)
            
            # 转换为Pydantic模型并返回JSON
            index_realtime = IndexRealTimeData(**index_data)
            return json.dumps(index_realtime.dict(), ensure_ascii=False, default=str)
            
        except Exception as e:
            error_msg = {"error": f"获取指数实时数据时发生错误: {str(e)}"}
            return json.dumps(error_msg, ensure_ascii=False)

    def get_multiple_stocks(self, symbols: List[str]) -> str:
        """
        获取多只股票的实时数据
        
        Args:
            symbols: 股票代码列表
        """
        try:
            results = []
            for symbol in symbols:
                if not symbol or len(symbol) != 6 or not symbol.isdigit():
                    results.append({
                        "symbol": symbol,
                        "error": "无效的股票代码"
                    })
                    continue
                
                stock_data = self._get_stock_data_from_api(symbol)
                if stock_data:
                    stock_realtime = StockRealTimeData(**stock_data)
                    results.append(stock_realtime.dict())
                else:
                    results.append({
                        "symbol": symbol,
                        "error": "获取数据失败"
                    })
            
            return json.dumps({"stocks": results}, ensure_ascii=False, default=str)
            
        except Exception as e:
            error_msg = {"error": f"获取多只股票数据时发生错误: {str(e)}"}
            return json.dumps(error_msg, ensure_ascii=False)

    def get_multiple_indices(self, symbols: List[str]) -> str:
        """
        获取多个指数的实时数据
        
        Args:
            symbols: 指数代码列表
        """
        try:
            results = []
            for symbol in symbols:
                if not symbol:
                    results.append({
                        "symbol": symbol,
                        "error": "无效的指数代码"
                    })
                    continue
                
                index_data = self._get_index_data_from_api(symbol)
                if index_data:
                    index_realtime = IndexRealTimeData(**index_data)
                    results.append(index_realtime.dict())
                else:
                    results.append({
                        "symbol": symbol,
                        "error": "获取数据失败"
                    })
            
            return json.dumps({"indices": results}, ensure_ascii=False, default=str)
            
        except Exception as e:
            error_msg = {"error": f"获取多个指数数据时发生错误: {str(e)}"}
            return json.dumps(error_msg, ensure_ascii=False)