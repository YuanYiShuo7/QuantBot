from typing import List
from datetime import timedelta

class ObservationConfig:
    """观测配置"""
    
    # 默认自选股票列表
    DEFAULT_WATCHLIST = ["000001", "000002", "600036", "601318", "000858"]
    
    # 默认观察的大盘指数
    DEFAULT_MARKET_INDICES = ["000001", "399001", "000300", "000905"]
    
    # 历史数据深度配置
    HISTORICAL_DEPTH = {
        "minute": timedelta(days=30),      # 30天的分钟数据
        "daily": timedelta(days=365),      # 1年的日数据
        "weekly": timedelta(days=365*2),   # 2年的周数据
        "monthly": timedelta(days=365*5),  # 5年的月数据
    }
    
    # 实时数据更新频率（秒）
    REALTIME_UPDATE_INTERVAL = 3
    
    # 观测数据缓存配置
    CACHE_SIZE = 1000

class MarketConfig:
    """市场配置"""
    
    # 交易时间
    TRADING_HOURS = {
        "morning_open": "09:30:00",
        "morning_close": "11:30:00",
        "afternoon_open": "13:00:00",
        "afternoon_close": "15:00:00"
    }
    
    # 支持的粒度
    SUPPORTED_GRANULARITIES = ["minute", "daily", "weekly", "monthly"]