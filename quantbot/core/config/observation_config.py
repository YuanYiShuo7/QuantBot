from typing import Dict, Any, List
from pydantic import BaseModel
from general.types.obervation_types import TimeGranularity

class ObservationConfig(BaseModel):
    """Observation配置类"""
    # 初始账户配置
    initial_cash: float = 100000.0
    initial_total_assets: float = 100000.0
    initial_net_assets: float = 100000.0
    
    # 初始持仓配置
    initial_positions: List[Dict[str, Any]] = []
    
    # 关注列表配置
    watch_list: List[str] = []
    market_index_list: List[str] = ["000001", "000300"]  # 默认关注上证指数和沪深300
    
    # 市场状态配置
    trading_enabled: bool = True
    market_status: str = "open"
    
    # K线数据条数配置 - 细粒度到日周月
    max_klines_count: Dict[TimeGranularity, int] = {
        TimeGranularity.DAILY: 10,
        TimeGranularity.WEEKLY: 10,
        TimeGranularity.MONTHLY: 10,
    }
    
    # 数据更新配置
    enable_historical_data: bool = True