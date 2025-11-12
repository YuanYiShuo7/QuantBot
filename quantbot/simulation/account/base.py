from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path
import pickle

from general.schemas.account_schema import AccountSchema, AccountData, PositionData
from general.schemas.order_schema import OrderSchema, OrderFormSchema, OrderResultSchema

from general.interfaces.account_interface import AccountInterface

class Account(AccountInterface):
    """默认账户接口实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 从配置获取参数
        self.initial_capital = self.config.get('initial_capital', 1000000.0)  # 默认100万
        self.watch_list = self.config.get('watch_list', [])
        self.market_index_list = self.config.get('market_index_list', [])
        self.start_timestamp = self.config.get('start_timestamp', datetime(2024, 1, 1))
        # 公共账户对象
        self.account = self._initialize_account()
        
        # 日志
        import logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"账户接口初始化完成")
        self.logger.info(f"初始资金: {self.initial_capital:,.2f}")
        self.logger.info(f"自选股: {self.watch_list}")
        self.logger.info(f"关注指数: {self.market_index_list}")
    
    def _initialize_account(self) -> AccountSchema:
        """初始化账户数据"""
        
        # 初始化账户信息
        account_info = AccountData(
            timestamp=self.start_timestamp,
            total_assets=self.initial_capital,
            net_assets=self.initial_capital,
            available_cash=self.initial_capital,
            market_value=0.0,
            total_profit_loss=0.0,
            total_profit_loss_rate=0.0,
            position_rate=0.0
        )
        
        return AccountSchema(
            timestamp=self.start_timestamp,
            account_info=account_info,
            positions=[],
            orders=[],
            watch_list=self.watch_list,
            market_index_list=self.market_index_list
        )
    
    def format_account_info_for_prompt(self) -> str:
        """
        将账户信息 AccountSchema 完整转换为 Agent 的 Prompt 文本
        """
        try:
            account = self.account
            
            prompt_lines = [
                "=== ACCOUNT DATA ===",
                f"Timestamp: {account.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            # 账户基本信息
            prompt_lines.append("=== ACCOUNT SUMMARY ===")
            ai = account.account_info
            prompt_lines.extend([
                f"Total Assets: {ai.total_assets:,.2f}",
                f"Net Assets: {ai.net_assets:,.2f}",
                f"Available Cash: {ai.available_cash:,.2f}",
                f"Market Value: {ai.market_value:,.2f}",
                f"Total P&L: {ai.total_profit_loss:+,.2f} ({ai.total_profit_loss_rate:+.4f}%)",
                f"Position Rate: {ai.position_rate:.4f}%",
                ""
            ])
            
            # 持仓信息
            prompt_lines.append("=== POSITIONS ===")
            if account.positions:
                for position in account.positions:
                    prompt_lines.extend([
                        f"Position: {position.symbol}({position.name})",
                        f"  Quantity: {position.quantity:,}",
                        f"  Available: {position.available_quantity:,}",
                        f"  Cost Price: {position.cost_price:.4f}",
                        f"  Current Price: {position.current_price:.4f}",
                        f"  Market Value: {position.market_value:,.2f}",
                        f"  P&L: {position.profit_loss:+,.2f} ({position.profit_loss_rate:+.4f}%)",
                        ""
                    ])
            else:
                prompt_lines.append("No positions")
                prompt_lines.append("")
            
            # 订单信息
            prompt_lines.append("=== ORDERS ===")
            if account.orders:
                for order in account.orders:
                    prompt_lines.extend([
                        f"Order: {order.order_id}",
                        f"  Symbol: {order.symbol}",
                        f"  Type: {order.order_type.value}",  # 使用 value 获取枚举值
                        f"  Status: {order.status.value}",    # 使用 value 获取枚举值
                        f"  Price: {order.price:.4f}",
                        f"  Quantity: {order.quantity:,}",
                        f"  Create Time: {order.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",  # 使用 timestamp 字段
                        ""
                    ])
            else:
                prompt_lines.append("No orders")
                prompt_lines.append("")
                
            # 自选股列表
            prompt_lines.append("=== WATCH LIST ===")
            if account.watch_list:
                for i, symbol in enumerate(account.watch_list, 1):
                    prompt_lines.append(f"{i}. {symbol}")
            else:
                prompt_lines.append("No watch list items")
            prompt_lines.append("")
            
            # 关注指数列表
            prompt_lines.append("=== MARKET INDEX LIST ===")
            if account.market_index_list:
                for i, index in enumerate(account.market_index_list, 1):
                    prompt_lines.append(f"{i}. {index}")
            else:
                prompt_lines.append("No market indices")
            prompt_lines.append("")
            
            return "\n".join(prompt_lines)
            
        except Exception as e:
            self.logger.error(f"格式化账户信息失败: {str(e)}")
            return f"Account data formatting failed: {str(e)}"