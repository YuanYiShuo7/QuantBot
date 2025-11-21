import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulate.market.base import Market

def test_market_simulation():
    """测试市场数据模拟"""
    
    # 配置参数
    config = {
        'cache_dir': 'quantbot\cache\market_cache',
        'watch_list': ['000858'],  # 五粮液
        'market_index_list': [],
        'start_timestamp': datetime(2024, 1, 2),
        'end_timestamp': datetime(2024, 1, 10),
        'daily_kline_days': 30,
        'weekly_kline_weeks': 12,
        'monthly_kline_months': 6
    }
    
    # 创建市场实例
    print("初始化市场接口...")
    market = Market(config)
    
    # 测试时间范围：2024年1月的前5个交易日
    start_timestamp = datetime(2024, 1, 2, 14, 45, 0)  # 2024-01-02 
    end_timestamp = datetime(2024, 1, 10, 15, 0, 0)    # 2024-01-10
    
    print(f"\n测试时间范围: {start_timestamp} 到 {end_timestamp}")
    print("=" * 60)
    
    # 模拟交易日循环
    current_timestamp = start_timestamp
    trading_days_simulated = 0
    
    while current_timestamp <= end_timestamp and trading_days_simulated < 5:
        print(f"\n📊 交易日 {trading_days_simulated + 1}: {current_timestamp.strftime('%Y-%m-%d %H:%M')}")
        print("-" * 50)
        
        try:
            # 更新市场数据
            market_schema = market.update_market_from_data_cache(current_timestamp)
            
            # 显示基本信息
            print(f"市场状态: {market_schema.market_status}")
            print(f"时间戳: {market_schema.timestamp}")
            
            prompt = market.format_market_info_for_prompt()
            print(f"\n📝 生成的Prompt长度: {len(prompt)} 字符")
            print("提示文本预览:")
            print(prompt)
            
            trading_days_simulated += 1
            
        except Exception as e:
            print(f"❌ 处理交易日 {current_timestamp} 时出错: {str(e)}")
        
        # 获取下一个交易日
        next_trading_day = market.get_next_trading_day(current_timestamp)
        if next_trading_day:
            current_timestamp = next_trading_day
        else:
            print("没有更多交易日数据")
            break
    
    print(f"\n🎯 模拟完成! 共处理 {trading_days_simulated} 个交易日")

if __name__ == "__main__":
    print("🚀 开始市场数据模拟测试")
    print("=" * 60)
    
    try:
        # 运行测试
        test_market_simulation()

    except Exception as e:
        print(f"❌ 测试执行失败: {str(e)}")
        import traceback
        traceback.print_exc()