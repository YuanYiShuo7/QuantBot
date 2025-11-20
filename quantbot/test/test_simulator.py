"""
Simulator 集成测试脚本
测试 Simulator 是否能正常初始化和运行
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulate.simulator.base import Simulator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_config():
    """创建测试配置"""
    # 测试时间范围：2024年1月的前5个交易日
    start_timestamp = datetime(2024, 1, 2, 14, 45, 0)  # 2024-01-02 是周二
    end_timestamp = datetime(2024, 1, 10, 15, 0, 0)    # 2024-01-10
    
    # 测试股票和指数（使用常见的A股代码）
    watch_list = ['000858']  # 五粮液
    market_index_list = ['000001']  # 上证指数
    
    # 使用 pathlib.Path 构建路径（支持使用 / 分隔符，自动处理跨平台兼容）
    # 转换为字符串时，Windows 会自动转换为 \，但使用 / 也不会报错
    script_dir = Path(__file__).parent.resolve()
    quantbot_dir = script_dir.parent
    
    config = {
        # 环境配置
        'environment': {
            'trading_interval_days': 1,  # 每天交易一次
        },
        
        # 计时器配置
        'timer': {
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
        },
        
        # 账户配置
        'account': {
            'initial_capital': 1000000.0,  # 100万初始资金
            'watch_list': watch_list,
            'market_index_list': market_index_list,
            'start_timestamp': start_timestamp,
        },
        
        # 市场配置
        'market': {
            'cache_dir': str(quantbot_dir / 'cache' / 'market_cache'),  # 使用 pathlib 构建路径
            'watch_list': watch_list,
            'market_index_list': market_index_list,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'daily_kline_days': 30,
            'weekly_kline_weeks': 12,
            'monthly_kline_months': 6,
        },
        
        # 交易所配置（使用默认值）
        'exchange': {
            'commission_rate': 0.0003,  # 万三
            'stamp_duty_rate': 0.001,   # 千一
            'transfer_fee_rate': 0.00002,
            'order_expiry_days': 7,
        },
        
        # LLM Agent配置（使用较小的模型或本地模型）
        'llm_agent': {
            'model_path': str(quantbot_dir / 'llm' / 'DeepSeek-V3'),  # 使用 pathlib 构建路径
            'device': 'cpu',  # 测试时使用CPU，避免GPU内存问题
            'temperature': 0.7,
            'max_new_tokens': 512,  # 减少生成长度以加快测试
        },
        
        # 奖励系统配置
        'reward': {
            'persist_path': str(quantbot_dir / 'trajectory' / 'test_trajectories.json'),  # 使用 pathlib 构建路径
        },
    }
    
    return config


def main():
    """主测试函数"""
    print("=" * 60)
    print("开始测试 Simulator")
    print("=" * 60)
    
    try:
        # 创建测试配置
        config = create_test_config()
        print("\n测试配置已创建")
        print(f"时间范围: {config['timer']['start_timestamp']} 到 {config['timer']['end_timestamp']}")
        print(f"测试股票: {config['account']['watch_list']}")
        print(f"测试指数: {config['account']['market_index_list']}")
        
        # 初始化 Simulator
        print("\n" + "-" * 60)
        print("初始化 Simulator...")
        print("-" * 60)
        simulator = Simulator(config)
        print("✓ Simulator 初始化成功")
        
        # 运行模拟
        print("\n" + "-" * 60)
        print("开始运行模拟...")
        print("-" * 60)
        simulator.run()
        print("✓ 模拟运行完成")
        
        # 检查奖励数据是否保存
        reward_path = config['reward']['persist_path']
        if os.path.exists(reward_path):
            print(f"\n✓ 奖励数据已保存至: {reward_path}")
        else:
            print(f"\n⚠ 警告: 奖励数据文件未找到: {reward_path}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

