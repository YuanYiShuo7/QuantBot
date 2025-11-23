from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests
import os
import sys
import akshare as ak

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
    prev_close: float = Field(..., description="前收盘价")

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
    prev_close: float = Field(..., description="前收盘价")
    
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
    
    # 振幅和换手率
    amplitude: float = Field(..., description="振幅")
    turnover_rate: float = Field(..., description="换手率")


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
            ],
            **kwargs
        )
        
        # 股票代码映射表（用于代码格式转换）
        self.stock_code_map = {
            "000001": "000001",  # 平安银行
            "000002": "000002",  # 万科A
            "000858": "000858",  # 五粮液
            "600519": "600519",  # 贵州茅台
            "601318": "601318",  # 中国平安
            "600036": "600036",  # 招商银行
            "000063": "000063",  # 中兴通讯
            "300059": "300059",  # 东方财富
            "002415": "002415",  # 海康威视
            "000333": "000333"   # 美的集团
        }
        
        # 指数代码映射表
        self.index_code_map = {
            "000001": "000001",  # 上证指数
            "399001": "399001",  # 深证成指
            "399006": "399006",  # 创业板指
            "000300": "000300",  # 沪深300
            "000905": "000905",  # 中证500
            "399005": "399005",  # 中小板指
            "000016": "000016",  # 上证50
            "399673": "399673"   # 创业板50
        }

    def _get_stock_data_from_akshare(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用akshare获取股票实时数据
        """
        try:
            # 获取实时行情数据
            stock_zh_a_spot_df = ak.stock_zh_a_spot_em()
            
            # 查找指定股票
            stock_data = stock_zh_a_spot_df[stock_zh_a_spot_df['代码'] == symbol]
            
            if stock_data.empty:
                # 如果没找到，尝试获取个股历史数据
                try:
                    stock_individual_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq")
                    if not stock_individual_df.empty:
                        latest_data = stock_individual_df.iloc[-1]
                        return {
                            "symbol": symbol,
                            "name": f"股票{symbol}",
                            "timestamp": datetime.now(),
                            "current_price": float(latest_data['收盘']),
                            "open": float(latest_data['开盘']),
                            "high": float(latest_data['最高']),
                            "low": float(latest_data['最低']),
                            "prev_close": float(latest_data['收盘']),
                            "bid_prices": [float(latest_data['收盘'])] * 5,
                            "bid_volumes": [0] * 5,
                            "ask_prices": [float(latest_data['收盘'])] * 5,
                            "ask_volumes": [0] * 5,
                            "volume": float(latest_data['成交量']),
                            "turnover": float(latest_data['成交额']),
                            "change": 0.0,
                            "change_rate": 0.0,
                            "amplitude": 0.0,
                            "turnover_rate": 0.0
                        }
                except:
                    pass
                return None
            
            # 提取数据
            stock_row = stock_data.iloc[0]
            
            # 构建盘口数据
            bid_prices = [
                float(stock_row['买一价']), float(stock_row['买二价']), 
                float(stock_row['买三价']), float(stock_row['买四价']), 
                float(stock_row['买五价'])
            ]
            bid_volumes = [
                float(stock_row['买一量']), float(stock_row['买二量']),
                float(stock_row['买三量']), float(stock_row['买四量']),
                float(stock_row['买五量'])
            ]
            ask_prices = [
                float(stock_row['卖一价']), float(stock_row['卖二价']),
                float(stock_row['卖三价']), float(stock_row['卖四价']),
                float(stock_row['卖五价'])
            ]
            ask_volumes = [
                float(stock_row['卖一量']), float(stock_row['卖二量']),
                float(stock_row['卖三量']), float(stock_row['卖四量']),
                float(stock_row['卖五量'])
            ]
            
            current_price = float(stock_row['最新价'])
            prev_close = float(stock_row['昨收'])
            change = current_price - prev_close
            change_rate = (change / prev_close * 100) if prev_close != 0 else 0
            
            return {
                "symbol": symbol,
                "name": stock_row['名称'],
                "timestamp": datetime.now(),
                "current_price": current_price,
                "open": float(stock_row['今开']),
                "high": float(stock_row['最高']),
                "low": float(stock_row['最低']),
                "prev_close": prev_close,
                "bid_prices": bid_prices,
                "bid_volumes": bid_volumes,
                "ask_prices": ask_prices,
                "ask_volumes": ask_volumes,
                "volume": float(stock_row['成交量']),
                "turnover": float(stock_row['成交额']),
                "change": round(change, 3),
                "change_rate": round(change_rate, 2),
                "amplitude": float(stock_row['振幅']),
                "turnover_rate": float(stock_row['换手率'])
            }
            
        except Exception as e:
            print(f"通过akshare获取股票数据时发生错误: {e}")
            return None

    def _get_index_data_from_akshare(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用akshare获取指数实时数据
        """
        try:
            # 获取指数实时行情
            index_spot_df = ak.stock_zh_index_spot_em()
            
            # 查找指定指数
            index_data = index_spot_df[index_spot_df['代码'] == symbol]
            
            if index_data.empty:
                return None
            
            # 提取数据
            index_row = index_data.iloc[0]
            
            current_price = float(index_row['最新价'])
            prev_close = float(index_row['昨收'])
            change = current_price - prev_close
            change_rate = (change / prev_close * 100) if prev_close != 0 else 0
            
            return {
                "symbol": symbol,
                "name": index_row['名称'],
                "timestamp": datetime.now(),
                "current_price": current_price,
                "change": round(change, 3),
                "change_rate": round(change_rate, 2),
                "volume": float(index_row['成交量']),
                "turnover": float(index_row['成交额']),
                "open": float(index_row['今开']),
                "high": float(index_row['最高']),
                "low": float(index_row['最低']),
                "prev_close": prev_close
            }
            
        except Exception as e:
            print(f"通过akshare获取指数数据时发生错误: {e}")
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
            
            # 获取真实数据
            stock_data = self._get_stock_data_from_akshare(symbol)
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
            
            # 获取真实数据
            index_data = self._get_index_data_from_akshare(symbol)
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
                
                stock_data = self._get_stock_data_from_akshare(symbol)
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
                
                index_data = self._get_index_data_from_akshare(symbol)
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