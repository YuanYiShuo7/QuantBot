from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import pickle
from pathlib import Path
import akshare as ak
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from general.schemas.market_schema import MarketSchema, StockData, IndexData, StockRealTimeData, IndexRealTimeData, KLineData, TimeGranularity
from general.interfaces.market_interface import MarketInterface

class Market(MarketInterface):
    """默认市场数据接口实现类 - 模拟模式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 从配置获取参数
        self.cache_dir = Path(self.config.get('cache_dir', './market_data_cache'))
        self.watch_list = self.config.get('watch_list', [])
        self.market_index_list = self.config.get('market_index_list', [])
        
        # 模拟模式配置
        self.start_timestamp = self.config.get('start_timestamp', datetime(2024, 1, 1))
        self.end_timestamp = self.config.get('end_timestamp', datetime(2024, 12, 31))
        self.step_interval = timedelta(days=1)  # 模拟模式按日更新
        
        # K线历史数据配置
        self.daily_kline_days = self.config.get('daily_kline_days', 30)
        self.weekly_kline_weeks = self.config.get('weekly_kline_weeks', 12)
        self.monthly_kline_months = self.config.get('monthly_kline_months', 6)
        
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # 私有市场对象
        self._market_schema = self._initialize_market_schema()
        
        # 数据缓存
        self._minute_data_cache: Dict[str, pd.DataFrame] = {}
        self._daily_data_cache: Dict[str, pd.DataFrame] = {}
        self._weekly_data_cache: Dict[str, pd.DataFrame] = {}
        self._monthly_data_cache: Dict[str, pd.DataFrame] = {}
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"模拟市场接口初始化完成")
        self.logger.info(f"时间范围: {self.start_timestamp} 到 {self.end_timestamp}")
        self.logger.info(f"关注股票: {self.watch_list}")
        self.logger.info(f"关注指数: {self.market_index_list}")
    
    def _initialize_market_schema(self) -> MarketSchema:
        """初始化市场数据"""
        return MarketSchema(
            timestamp=self.start_timestamp,
            market_status="closed",
            market_indices={},
            stock_data={}
        )
    
    def get_market_schema(self) -> MarketSchema:
        """获取当前市场数据的 MarketSchema 对象"""
        return self._market_schema
    
    def set_market_schema(self, market_schema: MarketSchema):
        """设置当前市场数据的 MarketSchema 对象"""
        self._market_schema = market_schema
    
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
        try:
            # 获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            stock_name = stock_info[stock_info['item'] == '股票简称']['value'].iloc[0]
            
            # 下载日线数据 - 根据日线条目数决定时间范围
            daily_start_date = (self.start_timestamp - timedelta(days=self.daily_kline_days)).strftime('%Y%m%d')
            daily_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            daily_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=daily_start_date,
                end_date=daily_end_date,
                adjust=""
            )
            self._daily_data_cache[symbol] = daily_data
            self._save_data_to_cache(symbol, 'daily', daily_data)
            
            # 下载周线数据 - 根据周线条目数决定时间范围
            weekly_start_date = (self.start_timestamp - timedelta(weeks=self.weekly_kline_weeks)).strftime('%Y%m%d')
            weekly_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            weekly_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="weekly",
                start_date=weekly_start_date,
                end_date=weekly_end_date,
                adjust=""
            )
            self._weekly_data_cache[symbol] = weekly_data
            self._save_data_to_cache(symbol, 'weekly', weekly_data)
            
            # 下载月线数据 - 根据月线条目数决定时间范围
            monthly_start_date = (self.start_timestamp - timedelta(days=self.monthly_kline_months * 30)).strftime('%Y%m%d')
            monthly_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            monthly_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="monthly",
                start_date=monthly_start_date,
                end_date=monthly_end_date,
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
            index_name = index_info[index_info['index_code'] == index]['display_name'].iloc[0]
            
            # 下载日线数据 - 根据日线条目数决定时间范围
            daily_start_date = (self.start_timestamp - timedelta(days=self.daily_kline_days)).strftime('%Y%m%d')
            daily_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            daily_data = ak.index_zh_a_hist(
                symbol=index,
                period="daily",
                start_date=daily_start_date,
                end_date=daily_end_date
            )
            self._daily_data_cache[index] = daily_data
            self._save_data_to_cache(index, 'daily', daily_data)
            
            # 下载周线数据 - 根据周线条目数决定时间范围
            weekly_start_date = (self.start_timestamp - timedelta(weeks=self.weekly_kline_weeks)).strftime('%Y%m%d')
            weekly_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            weekly_data = ak.index_zh_a_hist(
                symbol=index,
                period="weekly",
                start_date=weekly_start_date,
                end_date=weekly_end_date
            )
            self._weekly_data_cache[index] = weekly_data
            self._save_data_to_cache(index, 'weekly', weekly_data)
            
            # 下载月线数据 - 根据月线条目数决定时间范围
            monthly_start_date = (self.start_timestamp - timedelta(days=self.monthly_kline_months * 30)).strftime('%Y%m%d')
            monthly_end_date = self.end_timestamp.strftime('%Y%m%d')
            
            monthly_data = ak.index_zh_a_hist(
                symbol=index,
                period="monthly",
                start_date=monthly_start_date,
                end_date=monthly_end_date
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
    
    def update_market_from_data_cache(self, timestamp: datetime = None) -> MarketSchema:
        """
        从本地缓存加载市场数据并更新 MarketSchema
        """
        try:
            # 更新市场状态
            market_status = self._get_market_status(timestamp)
            self._market_schema.market_status = market_status
            self._market_schema.timestamp = timestamp
            
            # 更新股票数据
            for symbol in self.watch_list:
                stock_data = self._get_stock_data_at_timestamp(symbol, timestamp)
                if stock_data:
                    self._market_schema.stock_data[symbol] = stock_data
            
            # 更新指数数据
            for index in self.market_index_list:
                index_data = self._get_index_data_at_timestamp(index, timestamp)
                if index_data:
                    self._market_schema.market_indices[index] = index_data
            
            self.logger.debug(f"市场数据更新完成: {timestamp}")
            return self._market_schema
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {str(e)}")
            return self._market_schema
    
    def _get_market_status(self, timestamp: datetime) -> str:
        """获取市场状态"""
        # 判断交易时间：工作日 9:30-11:30, 13:00-15:00
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
        """获取股票实时数据 - 基于日K线数据模拟"""
        try:
            # 获取当天的日K线数据
            daily_data = self._get_daily_kline_data(symbol, timestamp, 1, is_index=False)
            if not daily_data:
                return None
            
            day_kline = daily_data[0]
            stock_name = self._get_stock_name(symbol)
            
            # 在模拟模式下，我们假设Agent看到的是交易日结束时的数据
            # 所以实时数据等于当日的收盘数据
            simulation_timestamp = timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
            
            # 模拟盘口数据 - 基于收盘价
            current_price = day_kline.close
            bid_prices = [round(current_price * 0.999, 2) for _ in range(5)]
            ask_prices = [round(current_price * 1.001, 2) for _ in range(5)]
            bid_volumes = [10000] * 5  # 模拟数据
            ask_volumes = [10000] * 5  # 模拟数据
            
            return StockRealTimeData(
                symbol=symbol,
                name=stock_name,
                timestamp=simulation_timestamp,
                current_price=current_price,
                open=day_kline.open,
                high=day_kline.high,
                low=day_kline.low,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                volume=day_kline.volume,
                turnover=day_kline.turnover,
                change=day_kline.change,
                change_rate=day_kline.change_rate
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
        """获取指数实时数据 - 基于日K线数据模拟"""
        try:
            # 获取当天的日K线数据
            daily_data = self._get_daily_kline_data(index, timestamp, 1, is_index=True)
            if not daily_data:
                return None
            
            day_kline = daily_data[0]
            index_name = self._get_index_name(index)
            
            # 在模拟模式下，我们假设Agent看到的是交易日结束时的数据
            simulation_timestamp = timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
            
            return IndexRealTimeData(
                symbol=index,
                name=index_name,
                timestamp=simulation_timestamp,
                current_price=day_kline.close,
                change=day_kline.change,
                change_rate=day_kline.change_rate,
                volume=day_kline.volume,
                turnover=day_kline.turnover,
                open=day_kline.open,
                high=day_kline.high,
                low=day_kline.low

            )
            
        except Exception as e:
            self.logger.error(f"获取指数 {index} 实时数据失败: {str(e)}")
            return None
    
    def _get_stock_kline_data(self, symbol: str, timestamp: datetime) -> Dict[str, List[KLineData]]:
        """获取股票K线数据 - 修正：始终返回可用的历史数据"""
        result = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        # 日线数据 - 始终返回，至少返回当天数据
        daily_data = self._get_daily_kline_data(symbol, timestamp, self.daily_kline_days, is_index=False)
        result['daily'] = daily_data
        
        # 周线数据 - 始终返回可用的周线数据，不检查时间间隔
        weekly_data = self._get_weekly_kline_data(symbol, timestamp, self.weekly_kline_weeks, is_index=False)
        result['weekly'] = weekly_data
        
        # 月线数据 - 始终返回可用的月线数据，不检查时间间隔
        monthly_data = self._get_monthly_kline_data(symbol, timestamp, self.monthly_kline_months, is_index=False)
        result['monthly'] = monthly_data
        
        return result
    
    def _get_index_kline_data(self, index: str, timestamp: datetime) -> Dict[str, List[KLineData]]:
        """获取指数K线数据 - 修正：始终返回可用的历史数据"""
        result = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
        # 日线数据 - 始终返回，至少返回当天数据
        daily_data = self._get_daily_kline_data(index, timestamp, self.daily_kline_days, is_index=True)
        result['daily'] = daily_data
        
        # 周线数据 - 始终返回可用的周线数据，不检查时间间隔
        weekly_data = self._get_weekly_kline_data(index, timestamp, self.weekly_kline_weeks, is_index=True)
        result['weekly'] = weekly_data
        
        # 月线数据 - 始终返回可用的月线数据，不检查时间间隔
        monthly_data = self._get_monthly_kline_data(index, timestamp, self.monthly_kline_months, is_index=True)
        result['monthly'] = monthly_data
        
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
            
            # 确保数据中有日期列
            if '日期' not in data.columns:
                return []
            
            # 将日期列统一转换为datetime对象
            data['date_dt'] = pd.to_datetime(data['日期'])
            
            # 过滤出截止到timestamp的数据
            target_date = pd.to_datetime(timestamp.date())  # 只比较日期部分
            filtered_data = data[data['date_dt'] <= target_date].tail(days)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.DAILY,
                    timestamp=row['date_dt'],  # 使用转换后的datetime对象
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
            
            # 确保数据中有日期列
            if '日期' not in data.columns:
                return []
            
            # 将日期列统一转换为datetime对象
            data['date_dt'] = pd.to_datetime(data['日期'])
            
            # 过滤出截止到timestamp的数据
            target_date = pd.to_datetime(timestamp.date())  # 只比较日期部分
            filtered_data = data[data['date_dt'] <= target_date].tail(weeks)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.WEEKLY,
                    timestamp=row['date_dt'],  # 使用转换后的datetime对象
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
            
            # 确保数据中有日期列
            if '日期' not in data.columns:
                return []
            
            # 将日期列统一转换为datetime对象
            data['date_dt'] = pd.to_datetime(data['日期'])
            
            # 过滤出截止到timestamp的数据
            target_date = pd.to_datetime(timestamp.date())  # 只比较日期部分
            filtered_data = data[data['date_dt'] <= target_date].tail(months)
            
            klines = []
            for _, row in filtered_data.iterrows():
                kline = KLineData(
                    symbol=symbol,
                    name=name,
                    granularity=TimeGranularity.MONTHLY,
                    timestamp=row['date_dt'],  # 使用转换后的datetime对象
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
        将市场信息 MarketSchema 完整转换为 Agent 的 Prompt 文本
        包含所有实时数据和K线数据，保持对称性和格式化
        """
        try:
            market = self._market_schema
            
            prompt_lines = [
                "=== MARKET DATA ===",
                f"Timestamp: {market.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Market Status: {market.market_status}",
                ""
            ]
            
            # 指数实时数据
            if market.market_indices:
                prompt_lines.append("=== INDEX REAL-TIME DATA ===")
                for index_code, index_data in market.market_indices.items():
                    rt = index_data.real_time
                    prompt_lines.extend([
                        f"Index: {index_code}({rt.name})",
                        f"  Current: {rt.current_price:.4f}",
                        f"  Change: {rt.change:+.4f} ({rt.change_rate:+.4f}%)",
                        f"  OHLC: {rt.open:.4f}/{rt.high:.4f}/{rt.low:.4f}/{rt.current_price:.4f}",
                        f"  Volume: {rt.volume:,.0f}",
                        f"  Turnover: {rt.turnover:,.2f}",
                        ""
                    ])
            
            # 股票实时数据
            if market.stock_data:
                prompt_lines.append("=== STOCK REAL-TIME DATA ===")
                for stock_code, stock_data in market.stock_data.items():
                    rt = stock_data.real_time
                    prompt_lines.extend([
                        f"Stock: {stock_code}({rt.name})",
                        f"  Current: {rt.current_price:.4f}",
                        f"  Change: {rt.change:+.4f} ({rt.change_rate:+.4f}%)",
                        f"  OHLC: {rt.open:.4f}/{rt.high:.4f}/{rt.low:.4f}/{rt.current_price:.4f}",
                        f"  Volume: {rt.volume:,.0f}",
                        f"  Turnover: {rt.turnover:,.2f}",
                        f"  Bid: {[f'{p:.4f}x{v:,.0f}' for p, v in zip(rt.bid_prices, rt.bid_volumes)]}",
                        f"  Ask: {[f'{p:.4f}x{v:,.0f}' for p, v in zip(rt.ask_prices, rt.ask_volumes)]}",
                        ""
                    ])
            
            # 指数K线数据
            if market.market_indices:
                prompt_lines.append("=== INDEX KLINE DATA ===")
                for index_code, index_data in market.market_indices.items():
                    # 日K线
                    if index_data.daily_klines:
                        prompt_lines.append(f"--- {index_code} DAILY KLINE (Latest {len(index_data.daily_klines)} periods) ---")
                        for kline in index_data.daily_klines[-5:]:  # 显示最近5条
                            prompt_lines.extend([
                                f"  Date: {kline.timestamp.strftime('%Y-%m-%d')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                f"    Amplitude: {kline.amplitude:.4f}%",
                                ""
                            ])
                    
                    # 周K线
                    if index_data.weekly_klines:
                        prompt_lines.append(f"--- {index_code} WEEKLY KLINE (Latest {len(index_data.weekly_klines)} periods) ---")
                        for kline in index_data.weekly_klines[-3:]:  # 显示最近3条
                            prompt_lines.extend([
                                f"  Week: {kline.timestamp.strftime('%Y-%m-%d')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                ""
                            ])
                    
                    # 月K线
                    if index_data.monthly_klines:
                        prompt_lines.append(f"--- {index_code} MONTHLY KLINE (Latest {len(index_data.monthly_klines)} periods) ---")
                        for kline in index_data.monthly_klines[-2:]:  # 显示最近2条
                            prompt_lines.extend([
                                f"  Month: {kline.timestamp.strftime('%Y-%m')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                ""
                            ])
            
            # 股票K线数据
            if market.stock_data:
                prompt_lines.append("=== STOCK KLINE DATA ===")
                for stock_code, stock_data in market.stock_data.items():
                    # 日K线
                    if stock_data.daily_klines:
                        prompt_lines.append(f"--- {stock_code} DAILY KLINE (Latest {len(stock_data.daily_klines)} periods) ---")
                        for kline in stock_data.daily_klines[-5:]:  # 显示最近5条
                            prompt_lines.extend([
                                f"  Date: {kline.timestamp.strftime('%Y-%m-%d')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                f"    Amplitude: {kline.amplitude:.4f}%, Turnover Rate: {kline.turnover_rate or 0:.4f}%",
                                ""
                            ])
                    
                    # 周K线
                    if stock_data.weekly_klines:
                        prompt_lines.append(f"--- {stock_code} WEEKLY KLINE (Latest {len(stock_data.weekly_klines)} periods) ---")
                        for kline in stock_data.weekly_klines[-3:]:  # 显示最近3条
                            prompt_lines.extend([
                                f"  Week: {kline.timestamp.strftime('%Y-%m-%d')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                f"    Turnover Rate: {kline.turnover_rate or 0:.4f}%",
                                ""
                            ])
                    
                    # 月K线
                    if stock_data.monthly_klines:
                        prompt_lines.append(f"--- {stock_code} MONTHLY KLINE (Latest {len(stock_data.monthly_klines)} periods) ---")
                        for kline in stock_data.monthly_klines[-2:]:  # 显示最近2条
                            prompt_lines.extend([
                                f"  Month: {kline.timestamp.strftime('%Y-%m')}",
                                f"    OHLC: {kline.open:.4f}/{kline.high:.4f}/{kline.low:.4f}/{kline.close:.4f}",
                                f"    Change: {kline.change:+.4f} ({kline.change_rate:+.4f}%)",
                                f"    Volume: {kline.volume:,.0f}, Turnover: {kline.turnover:,.2f}",
                                f"    Turnover Rate: {kline.turnover_rate or 0:.4f}%",
                                ""
                            ])
            
            return "\n".join(prompt_lines)
            
        except Exception as e:
            self.logger.error(f"格式化市场信息失败: {str(e)}")
            return f"Market data formatting failed: {str(e)}"
    
    def get_next_trading_day(self, current_timestamp: datetime) -> Optional[datetime]:
        """
        获取下一个交易日
        
        Args:
            current_timestamp: 当前时间戳
            
        Returns:
            下一个交易日的datetime对象，如果没有更多数据则返回None
        """
        try:
            # 使用第一个股票的数据作为交易日历参考
            if not self.watch_list:
                self.logger.warning("没有关注的股票，无法确定交易日历")
                return None
                
            symbol = self.watch_list[0]
            if symbol not in self._daily_data_cache:
                self.logger.warning(f"股票 {symbol} 的日线数据未缓存")
                return None
            
            daily_data = self._daily_data_cache[symbol]
            
            # 确保数据中有日期列
            if '日期' not in daily_data.columns:
                self.logger.warning("日线数据中没有日期列")
                return None
            
            # 将日期列转换为datetime对象
            daily_data = daily_data.copy()
            daily_data['date_dt'] = pd.to_datetime(daily_data['日期'])
            
            # 找到当前日期之后的下一个交易日
            current_date = pd.to_datetime(current_timestamp.date())
            future_days = daily_data[daily_data['date_dt'] > current_date]
            
            if future_days.empty:
                self.logger.info("没有更多的交易日数据")
                return None
            
            # 返回下一个交易日
            next_trading_day = future_days.iloc[0]['date_dt']
            
            # 总是设置为决策时间（14:45）
            next_timestamp = datetime(
                year=next_trading_day.year,
                month=next_trading_day.month,
                day=next_trading_day.day,
                hour=14,  # 固定为14:45
                minute=45,
                second=0,
                microsecond=0
            )
            
            self.logger.debug(f"下一个交易日: {next_timestamp}")
            return next_timestamp
            
        except Exception as e:
            self.logger.error(f"获取下一个交易日失败: {str(e)}")
            return None