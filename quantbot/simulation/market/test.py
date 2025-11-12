import logging
from datetime import datetime, timedelta
from base import Market
import os
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def simulate_market_trading():
    """模拟市场交易演示"""
    print("=== 模拟市场交易演示 ===\n")
    
    # 配置模拟市场
    config = {
        'start_timestamp': datetime(2024, 6, 1),
        'end_timestamp': datetime(2024, 6, 30),
        'watch_list': ['000858'],  # 贵州茅台
        'market_index_list': ['000001'],     # 上证指数
        'cache_dir': './market_cache',
        'daily_kline_days': 30,
        'weekly_kline_weeks': 3,
        'monthly_kline_months': 3
    }
    
    # 创建市场实例
    market = Market(config)
    
    # 初始化数据缓存
    print("1. 初始化市场数据缓存...")
    success = market.initialize_market_data_cache()
    if not success:
        print("数据缓存初始化失败，退出演示")
        return
    
    print("数据缓存初始化成功！")
    
    # 显示缓存文件
    print("\n2. 缓存文件列表:")
    cache_dir = Path('./market_cache')
    if cache_dir.exists():
        for file in cache_dir.glob('*'):
            print(f"  - {file.name}")
    
    # 模拟交易过程
    print("\n3. 开始模拟交易...")
    
    # 重置到起始时间
    market.reset_to_start()
    
    # 模拟5个交易日
    for day in range(5):
        if not market.has_next_data():
            break
            
        print(f"\n--- 第{day+1}个交易日 ---")
        
        # 前进到下一个交易日
        market_data = market.update_market_from_data_cache(timestamp=datetime(2024, 6, 1) + timedelta(days=day))
        
        # 显示市场信息
        market_info = market.format_market_info_for_prompt()
        print(market_info)
        
        # 显示详细的K线数据
        if market_data.stock_data:
            for symbol, stock_data in market_data.stock_data.items():
                if stock_data.daily_klines:
                    latest_kline = stock_data.daily_klines[-1]
                    print(f"\n{symbol} 最新日K线:")
                    print(f"  日期: {latest_kline.timestamp.strftime('%Y-%m-%d')}")
                    print(f"  开盘: {latest_kline.open:.2f}, 最高: {latest_kline.high:.2f}, 最低: {latest_kline.low:.2f}, 收盘: {latest_kline.close:.2f}")
                    print(f"  成交量: {latest_kline.volume:,}, 成交额: {latest_kline.turnover:,.0f}")
                    print(f"  涨跌幅: {latest_kline.change_rate:+.2f}%")
    
    print("\n=== 模拟交易完成 ===")

def check_cached_data():
    """检查缓存的数据"""
    print("\n=== 检查缓存数据 ===")
    
    cache_dir = Path('./market_cache')
    if not cache_dir.exists():
        print("缓存目录不存在")
        return
    
    # 显示所有缓存文件
    print("缓存文件:")
    for file in cache_dir.glob('*'):
        file_size = file.stat().st_size
        print(f"  {file.name} ({file_size} bytes)")
    
    # 显示名称映射
    stock_names_file = cache_dir / "stock_names.json"
    if stock_names_file.exists():
        import json
        with open(stock_names_file, 'r', encoding='utf-8') as f:
            stock_names = json.load(f)
        print(f"\n股票名称映射: {stock_names}")
    
    index_names_file = cache_dir / "index_names.json"
    if index_names_file.exists():
        import json
        with open(index_names_file, 'r', encoding='utf-8') as f:
            index_names = json.load(f)
        print(f"指数名称映射: {index_names}")

if __name__ == "__main__":
    # 运行模拟交易
    simulate_market_trading()
    
    # 检查缓存数据
    check_cached_data()