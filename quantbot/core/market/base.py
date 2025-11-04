from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import pickle
from pathlib import Path
import akshare as ak
import logging
from enum import Enum
from pydantic import BaseModel, Field

from general.schemas.market_schema import MarketSchema, StockData, IndexData, StockRealTimeData, IndexRealTimeData, KLineData, TimeGranularity
from general.interfaces.market_interface import MarketInterface

class Market(MarketInterface):
    """默认市场数据接口实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 从配置获取参数
        self.cache_dir = Path(self.config.get('cache_dir', './market_data_cache'))
        self.watch_list = self.config.get('watch_list', [])
        self.market_index_list = self.config.get('market_index_list', [])
        self.start_timestamp = self.config.get('start_timestamp', datetime(2024, 1, 1))
        self.end_timestamp = self.config.get('end_timestamp', datetime(2024, 12, 31))
        self.step_interval = self.config.get('step_interval', timedelta(minutes=1))
        
        # K线历史数据配置
        self.daily_kline_days = self.config.get('daily_kline_days', 30)  # 默认30天日线
        self.weekly_kline_weeks = self.config.get('weekly_kline_weeks', 12)  # 默认12周周线
        self.monthly_kline_months = self.config.get('monthly_kline_months', 6)  # 默认6月月线
        
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # 公共市场对象
        self.market = self._initialize_market()
        
        # 数据缓存
        self._minute_data_cache: Dict[str, pd.DataFrame] = {}
        self._daily_data_cache: Dict[str, pd.DataFrame] = {}
        self._weekly_data_cache: Dict[str, pd.DataFrame] = {}
        self._monthly_data_cache: Dict[str, pd.DataFrame] = {}
        
        # 当前时间指针
        self._current_timestamp = self.start_timestamp
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"市场接口初始化完成，时间范围: {self.start_timestamp} 到 {self.end_timestamp}")
        self.logger.info(f"关注股票: {self.watch_list}")
        self.logger.info(f"关注指数: {self.market_index_list}")
    
    def _initialize_market(self) -> MarketSchema:
        """初始化市场数据"""
        return MarketSchema(
            timestamp=self.start_timestamp,
            market_status="closed",  # 初始状态为闭市
            market_indices={},
            stock_data={}
        )
    
    def initialize_market_data_cache(self) -> bool:
        """
        初始化数据缓存 - 通过akshare下载历史数据并存储在本地
        """
        try:
            self.logger.info("开始初始化市场数据缓存...")
            
            # 下载股票数据
            for symbol in self.watch_list:
                try:
                    self._download_stock_data(symbol)
                    self.logger.info(f"股票 {symbol} 数据下载完成")
                except Exception as e:
                    self.logger.error(f"股票 {symbol} 数据下载失败: {str(e)}")
            
            # 下载指数数据
            for index in self.market_index_list:
                try:
                    self._download_index_data(index)
                    self.logger.info(f"指数 {index} 数据下载完成")
                except Exception as e:
                    self.logger.error(f"指数 {index} 数据下载失败: {str(e)}")
            
            self.logger.info("市场数据缓存初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"市场数据缓存初始化失败: {str(e)}")
            return False
    
    def _download_stock_data(self, symbol: str) -> None:
        """下载股票数据"""
        # 分钟级数据（实时数据）
        try:
            # 获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            stock_name = stock_info[stock_info['item'] == '股票简称']['value'].iloc[0]
            
            # 下载分钟级数据
            minute_data = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=self.start_timestamp.strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d'),
                period='1',  # 1分钟
                adjust=''
            )
            self._minute_data_cache[symbol] = minute_data
            self._save_data_to_cache(symbol, 'minute', minute_data)
            
            # 下载日线数据
            daily_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=(self.start_timestamp - timedelta(days=self.daily_kline_days)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d'),
                adjust=""
            )
            self._daily_data_cache[symbol] = daily_data
            self._save_data_to_cache(symbol, 'daily', daily_data)
            
            # 下载周线数据
            weekly_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="weekly",
                start_date=(self.start_timestamp - timedelta(weeks=self.weekly_kline_weeks)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d'),
                adjust=""
            )
            self._weekly_data_cache[symbol] = weekly_data
            self._save_data_to_cache(symbol, 'weekly', weekly_data)
            
            # 下载月线数据
            monthly_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="monthly",
                start_date=(self.start_timestamp - timedelta(days=self.monthly_kline_months*30)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d'),
                adjust=""
            )
            self._monthly_data_cache[symbol] = monthly_data
            self._save_data_to_cache(symbol, 'monthly', monthly_data)
            
            # 保存股票名称映射
            self._save_stock_name_mapping(symbol, stock_name)
            
        except Exception as e:
            self.logger.error(f"下载股票 {symbol} 数据失败: {str(e)}")
            raise
    
    def _download_index_data(self, index: str) -> None:
        """下载指数数据"""
        try:
            # 获取指数名称
            index_info = ak.index_stock_info()
            index_name = index_info[index_info['code'] == index]['display_name'].iloc[0] if not index_info.empty else f"指数{index}"
            
            # 下载分钟级数据
            minute_data = ak.stock_zh_index_hist_min_em(
                symbol=index,
                start_date=self.start_timestamp.strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d'),
                period='1'
            )
            self._minute_data_cache[index] = minute_data
            self._save_data_to_cache(index, 'minute', minute_data)
            
            # 下载日线数据
            daily_data = ak.index_zh_a_hist(
                symbol=index,
                period="daily",
                start_date=(self.start_timestamp - timedelta(days=self.daily_kline_days)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d')
            )
            self._daily_data_cache[index] = daily_data
            self._save_data_to_cache(index, 'daily', daily_data)
            
            # 下载周线数据
            weekly_data = ak.index_zh_a_hist(
                symbol=index,
                period="weekly",
                start_date=(self.start_timestamp - timedelta(weeks=self.weekly_kline_weeks)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d')
            )
            self._weekly_data_cache[index] = weekly_data
            self._save_data_to_cache(index, 'weekly', weekly_data)
            
            # 下载月线数据
            monthly_data = ak.index_zh_a_hist(
                symbol=index,
                period="monthly",
                start_date=(self.start_timestamp - timedelta(days=self.monthly_kline_months*30)).strftime('%Y%m%d'),
                end_date=self.end_timestamp.strftime('%Y%m%d')
            )
            self._monthly_data_cache[index] = monthly_data
            self._save_data_to_cache(index, 'monthly', monthly_data)
            
            # 保存指数名称映射
            self._save_index_name_mapping(index, index_name)
            
        except Exception as e:
            self.logger.error(f"下载指数 {index} 数据失败: {str(e)}")
            raise
    
    def _save_data_to_cache(self, symbol: str, data_type: str, data: pd.DataFrame) -> None:
        """保存数据到缓存文件"""
        cache_file = self.cache_dir / f"{symbol}_{data_type}.pkl"
        data.to_pickle(cache_file)
    
    def _save_stock_name_mapping(self, symbol: str, name: str) -> None:
        """保存股票名称映射"""
        mapping_file = self.cache_dir / "stock_names.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
        else:
            mappings = {}
        
        mappings[symbol] = name
        
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    def _save_index_name_mapping(self, index: str, name: str) -> None:
        """保存指数名称映射"""
        mapping_file = self.cache_dir / "index_names.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
        else:
            mappings = {}
        
        mappings[index] = name
        
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    def update_market_from_data_cache(self, timestamp: datetime) -> MarketSchema:
        """
        从本地缓存加载市场数据并更新 MarketSchema
        """
        try:
            self._current_timestamp = timestamp
            
            # 更新市场状态
            market_status = self._get_market_status(timestamp)
            self.market.market_status = market_status
            self.market.timestamp = timestamp
            
            # 更新股票数据
            for symbol in self.watch_list:
                stock_data = self._get_stock_data_at_timestamp(symbol, timestamp)
                if stock_data:
                    self.market.stock_data[symbol] = stock_data
            
            # 更新指数数据
            for index in self.market_index_list:
                index_data = self._get_index_data_at_timestamp(index, timestamp)
                if index_data:
                    self.market.market_indices[index] = index_data
            
            self.logger.debug(f"市场数据更新完成: {timestamp}")
            return self.market
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {str(e)}")
            return self.market
    
    def _get_market_status(self, timestamp: datetime) -> str:
        """获取市场状态"""
        # 简单判断交易时间：工作日 9:30-11:30, 13:00-15:00
        if timestamp.weekday() >= 5:  # 周末
            return "closed"
        
        time_str = timestamp.strftime('%H:%M')
        if ('09:30' <= time_str <= '11:30') or ('13:00' <= time_str <= '15:00'):
            return "open"
        else:
            return "closed"
    
    def _get_stock_data_at_timestamp(self, symbol: str, timestamp: datetime) -> Optional[StockData]:
        """获取指定时间点的股票数据"""
        try:
            # 获取实时数据
            real_time_data = self._get_stock_real_time_data(symbol, timestamp)
            if not real_time_data:
                return None
            
            # 获取K线数据
            kline_data = self._get_stock_kline_data(symbol, timestamp)
            
            return StockData(
                real_time=real_time_data,
                daily_klines=kline_data['daily'],
                weekly_klines=kline_data['weekly'],
                monthly_klines=kline_data['monthly']
            )
            
        except Exception as e:
            self.logger.error(f"获取股票 {symbol} 数据失败: {str(e)}")
            return None
    
    def _get_stock_real_time_data(self, symbol: str, timestamp: datetime) -> Optional[StockRealTimeData]:
        """获取股票实时数据"""
        try:
            if symbol not in self._minute_data_cache:
                # 尝试从缓存文件加载
                cache_file = self.cache_dir / f"{symbol}_minute.pkl"
                if cache_file.exists():
                    self._minute_data_cache[symbol] = pd.read_pickle(cache_file)
                else:
                    return None
            
            minute_data = self._minute_data_cache[symbol]
            
            # 找到最接近的时间点
            target_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            if '日期' in minute_data.columns and '时间' in minute_data.columns:
                minute_data['datetime'] = pd.to_datetime(minute_data['日期'] + ' ' + minute_data['时间'])
            elif 'datetime' in minute_data.columns:
                minute_data['datetime'] = pd.to_datetime(minute_data['datetime'])
            
            closest_row = self._find_closest_timestamp(minute_data, timestamp)
            if closest_row is None:
                return None
            
            # 获取股票名称
            stock_name = self._get_stock_name(symbol)
            
            # 构建实时数据对象
            return StockRealTimeData(
                symbol=symbol,
                name=stock_name,
                timestamp=timestamp,
                current_price=float(closest_row['收盘']),
                open=float(closest_row['开盘']),
                high=float(closest_row['最高']),
                low=float(closest_row['最低']),
                pre_close=float(closest_row.get('前收盘', closest_row['收盘'])),
                change=float(closest_row.get('涨跌额', 0)),
                change_rate=float(closest_row.get('涨跌幅', 0)),
                volume=int(closest_row.get('成交量', 0)),
                turnover=float(closest_row.get('成交额', 0)),
                bid_prices=[float(closest_row.get('买一价', 0))] * 5,  # 简化处理
                bid_volumes=[int(closest_row.get('买一量', 0))] * 5,
                ask_prices=[float(closest_row.get('卖一价', 0))] * 5,
                ask_volumes=[int(closest_row.get('卖一量', 0))] * 5,
                pe_ratio=None,  # 需要额外接口获取
                pb_ratio=None,
                amplitude=float(closest_row.get('振幅', 0)),
                turnover_rate=float(closest_row.get('换手率', 0)),
                volume_ratio=None,
                committee=None,
                main_net_inflow=None,
                large_net_inflow=None,
                medium_net_inflow=None,
                small_net_inflow=None
            )
            
        except Exception as e:
            self.logger.error(f"获取股票 {symbol} 实时数据失败: {str(e)}")
            return None
    
    def _get_index_data_at_timestamp(self, index: str, timestamp: datetime) -> Optional[IndexData]:
        """获取指定时间点的指数数据"""
        try:
            # 获取实时数据
            real_time_data = self._get_index_real_time_data(index, timestamp)
            if not real_time_data:
                return None
            
            # 获取K线数据
            kline_data = self._get_index_kline_data(index, timestamp)
            
            return IndexData(
                real_time=real_time_data,
                daily_klines=kline_data['daily'],
                weekly_klines=kline_data['weekly'],
                monthly_klines=kline_data['monthly']
            )
            
        except Exception as e:
            self.logger.error(f"获取指数 {index} 数据失败: {str(e)}")
            return None
    
    def _get_index_real_time_data(self, index: str, timestamp: datetime) -> Optional[IndexRealTimeData]:
        """获取指数实时数据"""
        try:
            if index not in self._minute_data_cache:
                # 尝试从缓存文件加载
                cache_file = self.cache_dir / f"{index}_minute.pkl"
                if cache_file.exists():
                    self._minute_data_cache[index] = pd.read_pickle(cache_file)
                else:
                    return None
            
            minute_data = self._minute_data_cache[index]
            
            # 找到最接近的时间点
            closest_row = self._find_closest_timestamp(minute_data, timestamp)
            if closest_row is None:
                return None
            
            # 获取指数名称
            index_name = self._get_index_name(index)
            
            # 构建实时数据对象
            return IndexRealTimeData(
                symbol=index,
                name=index_name,
                timestamp=timestamp,
                current_price=float(closest_row['收盘']),
                change=float(closest_row.get('涨跌额', 0)),
                change_rate=float(closest_row.get('涨跌幅', 0)),
                volume=float(closest_row.get('成交量', 0)),
                turnover=float(closest_row.get('成交额', 0)),
                open=float(closest_row['开盘']),
                high=float(closest_row['最高']),
                low=float(closest_row['最低']),
                pre_close=float(closest_row.get('前收盘', closest_row['收盘'])),
                pe_ratio=None,
                pb_ratio=None,
                amplitude=float(closest_row.get('振幅', 0)),
                turnover_rate=None
            )
            
        except Exception as e:
            self.logger.error(f"获取指数 {index} 实时数据失败: {str(e)}")
            return None
    
    def _get_stock_kline_data(self, symbol: str, timestamp: datetime) -> Dict[str, List[KLineData]]:
        """获取股票K线数据"""
        result = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        # 计算时间间隔
        time_from_start = timestamp - self.start_timestamp
        
        # 日线数据（如果超过一天）
        if time_from_start >= timedelta(days=1):
            result['daily'] = self._get_daily_kline_data(symbol, timestamp, self.daily_kline_days)
        
        # 周线数据（如果超过一周）
        if time_from_start >= timedelta(weeks=1):
            result['weekly'] = self._get_weekly_kline_data(symbol, timestamp, self.weekly_kline_weeks)
        
        # 月线数据（如果超过一月）
        if time_from_start >= timedelta(days=30):
            result['monthly'] = self._get_monthly_kline_data(symbol, timestamp, self.monthly_kline_months)
        
        return result
    
    def _get_index_kline_data(self, index: str, timestamp: datetime) -> Dict[str, List[KLineData]]:
        """获取指数K线数据"""
        result = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        # 计算时间间隔
        time_from_start = timestamp - self.start_timestamp
        
        # 日线数据（如果超过一天）
        if time_from_start >= timedelta(days=1):
            result['daily'] = self._get_daily_kline_data(index, timestamp, self.daily_kline_days, is_index=True)
        
        # 周线数据（如果超过一周）
        if time_from_start >= timedelta(weeks=1):
            result['weekly'] = self._get_weekly_kline_data(index, timestamp, self.weekly_kline_weeks, is_index=True)
        
        # 月线数据（如果超过一月）
        if time_from_start >= timedelta(days=30):
            result['monthly'] = self._get_monthly_kline_data(index, timestamp, self.monthly_kline_months, is_index=True)
        
        return result
    
    def _get_daily_kline_data(self, symbol: str, timestamp: datetime, days: int, is_index: bool = False) -> List[KLineData]:
        """获取日K线数据"""
        try:
            cache_key = symbol
            data_type = 'daily'
            
            if cache_key not in self._daily_data_cache:
                cache_file = self.cache_dir / f"{symbol}_{data_type}.pkl"
                if cache_file.exists():
                    self._daily_data_cache[cache_key] = pd.read_pickle(cache_file)
                else:
                    return []
            
            data = self._daily_data_cache[cache_key]
            name = self._get_index_name(symbol) if is_index else self._get_stock_name(symbol)
            
            # 过滤出截止到timestamp的数据
            end_date = timestamp.strftime('%Y-%m-%d')
            if '日期' in data.columns:
                filtered_data = data[data['日期'] <= end_date].tail(days)
            else:
                filtered_data = data.tail(days)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.DAILY,
                    timestamp=pd.to_datetime(row['日期']),
                    open=float(row['开盘']),
                    high=float(row['最高']),
                    low=float(row['最低']),
                    close=float(row['收盘']),
                    volume=int(row.get('成交量', 0)),
                    turnover=float(row.get('成交额', 0)),
                    change=float(row.get('涨跌额', 0)),
                    change_rate=float(row.get('涨跌幅', 0)),
                    amplitude=float(row.get('振幅', 0)),
                    turnover_rate=float(row.get('换手率', 0)) if not is_index else None
                )
                klines.append(kline)
            
            return klines
            
        except Exception as e:
            self.logger.error(f"获取{symbol}日K线数据失败: {str(e)}")
            return []
    
    def _get_weekly_kline_data(self, symbol: str, timestamp: datetime, weeks: int, is_index: bool = False) -> List[KLineData]:
        """获取周K线数据"""
        try:
            cache_key = symbol
            data_type = 'weekly'
            
            if cache_key not in self._weekly_data_cache:
                cache_file = self.cache_dir / f"{symbol}_{data_type}.pkl"
                if cache_file.exists():
                    self._weekly_data_cache[cache_key] = pd.read_pickle(cache_file)
                else:
                    return []
            
            data = self._weekly_data_cache[cache_key]
            name = self._get_index_name(symbol) if is_index else self._get_stock_name(symbol)
            
            # 过滤出截止到timestamp的数据
            end_date = timestamp.strftime('%Y-%m-%d')
            if '日期' in data.columns:
                filtered_data = data[data['日期'] <= end_date].tail(weeks)
            else:
                filtered_data = data.tail(weeks)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.WEEKLY,
                    timestamp=pd.to_datetime(row['日期']),
                    open=float(row['开盘']),
                    high=float(row['最高']),
                    low=float(row['最低']),
                    close=float(row['收盘']),
                    volume=int(row.get('成交量', 0)),
                    turnover=float(row.get('成交额', 0)),
                    change=float(row.get('涨跌额', 0)),
                    change_rate=float(row.get('涨跌幅', 0)),
                    amplitude=float(row.get('振幅', 0)),
                    turnover_rate=float(row.get('换手率', 0)) if not is_index else None
                )
                klines.append(kline)
            
            return klines
            
        except Exception as e:
            self.logger.error(f"获取{symbol}周K线数据失败: {str(e)}")
            return []
    
    def _get_monthly_kline_data(self, symbol: str, timestamp: datetime, months: int, is_index: bool = False) -> List[KLineData]:
        """获取月K线数据"""
        try:
            cache_key = symbol
            data_type = 'monthly'
            
            if cache_key not in self._monthly_data_cache:
                cache_file = self.cache_dir / f"{symbol}_{data_type}.pkl"
                if cache_file.exists():
                    self._monthly_data_cache[cache_key] = pd.read_pickle(cache_file)
                else:
                    return []
            
            data = self._monthly_data_cache[cache_key]
            name = self._get_index_name(symbol) if is_index else self._get_stock_name(symbol)
            
            # 过滤出截止到timestamp的数据
            end_date = timestamp.strftime('%Y-%m-%d')
            if '日期' in data.columns:
                filtered_data = data[data['日期'] <= end_date].tail(months)
            else:
                filtered_data = data.tail(months)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.MONTHLY,
                    timestamp=pd.to_datetime(row['日期']),
                    open=float(row['开盘']),
                    high=float(row['最高']),
                    low=float(row['最低']),
                    close=float(row['收盘']),
                    volume=int(row.get('成交量', 0)),
                    turnover=float(row.get('成交额', 0)),
                    change=float(row.get('涨跌额', 0)),
                    change_rate=float(row.get('涨跌幅', 0)),
                    amplitude=float(row.get('振幅', 0)),
                    turnover_rate=float(row.get('换手率', 0)) if not is_index else None
                )
                klines.append(kline)
            
            return klines
            
        except Exception as e:
            self.logger.error(f"获取{symbol}月K线数据失败: {str(e)}")
            return []
    
    def _find_closest_timestamp(self, data: pd.DataFrame, timestamp: datetime) -> Optional[pd.Series]:
        """在数据框中找到最接近时间戳的行"""
        if 'datetime' not in data.columns:
            return None
        
        # 找到小于等于目标时间戳的最新数据
        mask = data['datetime'] <= timestamp
        if not mask.any():
            return None
        
        filtered_data = data[mask]
        if filtered_data.empty:
            return None
        
        return filtered_data.iloc[-1]
    
    def _get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        mapping_file = self.cache_dir / "stock_names.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
                return mappings.get(symbol, f"股票{symbol}")
        return f"股票{symbol}"
    
    def _get_index_name(self, index: str) -> str:
        """获取指数名称"""
        mapping_file = self.cache_dir / "index_names.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
                return mappings.get(index, f"指数{index}")
        return f"指数{index}"
    
    def format_market_info_for_prompt(self) -> str:
        """
        将市场信息 MarketSchema 转换为 Agent 的 Prompt 文本
        """
        try:
            market = self.market
            
            prompt_lines = [
                "=== 市场信息 ===",
                f"更新时间: {market.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"市场状态: {market.market_status}",
                ""
            ]
            
            # 指数信息
            if market.market_indices:
                prompt_lines.append("=== 大盘指数 ===")
                for index_code, index_data in market.market_indices.items():
                    rt = index_data.real_time
                    prompt_lines.append(
                        f"{index_code}({rt.name}): {rt.current_price:.2f} "
                        f"{rt.change:+.2f}({rt.change_rate:+.2f}%) "
                        f"振幅:{rt.amplitude:.2f}% 成交量:{rt.volume:,.0f}"
                    )
                prompt_lines.append("")
            
            # 股票信息
            if market.stock_data:
                prompt_lines.append("=== 股票行情 ===")
                prompt_lines.append(f"{'代码':<8} {'名称':<10} {'现价':>8} {'涨跌':>8} {'涨跌幅':>8} {'成交量':>12} {'成交额':>12}")
                prompt_lines.append("-" * 80)
                
                for stock_code, stock_data in market.stock_data.items():
                    rt = stock_data.real_time
                    prompt_lines.append(
                        f"{stock_code:<8} {rt.name:<10} {rt.current_price:>8.2f} "
                        f"{rt.change:>+8.2f} {rt.change_rate:>+7.2f}% "
                        f"{rt.volume:>12,d} {rt.turnover:>12,.0f}"
                    )
                
                # 添加盘口信息示例
                if len(market.stock_data) > 0:
                    first_stock = list(market.stock_data.values())[0].real_time
                    prompt_lines.append("")
                    prompt_lines.append("=== 盘口信息示例 ===")
                    prompt_lines.append(f"买盘: {first_stock.bid_prices[0]:.2f}×{first_stock.bid_volumes[0]:,} ...")
                    prompt_lines.append(f"卖盘: {first_stock.ask_prices[0]:.2f}×{first_stock.ask_volumes[0]:,} ...")
            
            return "\n".join(prompt_lines)
            
        except Exception as e:
            self.logger.error(f"格式化市场信息失败: {str(e)}")
            return f"市场信息格式化失败: {str(e)}"
    
    def get_next_timestamp(self) -> datetime:
        """获取下一个时间戳"""
        return self._current_timestamp + self.step_interval
    
    def has_next_data(self) -> bool:
        """检查是否还有后续数据"""
        return self._current_timestamp < self.end_timestamp
    
    def reset_to_start(self) -> None:
        """重置到起始时间"""
        self._current_timestamp = self.start_timestamp
        self.market = self._initialize_market()
        self.logger.info("市场数据已重置到起始时间")