from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libs.agno.tools import Toolkit
from libs.agno.run import RunContext

from general.schemas.account_schema import AccountSchema, AccountData, PositionData

class WatchListTools(Toolkit):
    """自选股工具类"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="watch_list_tools",
            tools=[
                self.add_watch_stock,
                self.remove_watch_stock,
                self.add_market_index,
                self.remove_market_index,
            ],
            **kwargs
        )

    def _get_account(self, run_context: RunContext) -> Optional[AccountSchema]:
        """从session_state获取并解析account"""
        try:
            account_dict = run_context.session_state.get("account_dict")
            if not account_dict:
                return None
            return AccountSchema(**account_dict)
        except Exception as e:
            print(f"解析account时发生错误: {str(e)}")
            return None

    def _save_account(self, run_context: RunContext, account: AccountSchema) -> bool:
        """保存account到session_state"""
        try:
            account_dict = account.dict()
            run_context.session_state["account_dict"] = account_dict
            return True
        except Exception as e:
            print(f"保存account时发生错误: {str(e)}")
            return False

    def add_watch_stock(self, run_context: RunContext, symbol: str) -> str:
        """
        添加自选股
        
        Args:
            symbol: 股票代码，6位数字
        """
        try:
            # 验证股票代码
            if not symbol or len(symbol) != 6 or not symbol.isdigit():
                return "错误：股票代码必须为6位数字"
            
            account = self._get_account(run_context)
            if not account:
                return "错误：未找到账户信息"
            
            # 初始化自选股列表
            if not account.watch_list:
                account.watch_list = []
            
            # 检查是否已存在
            if symbol in account.watch_list:
                return f"股票 {symbol} 已在自选股列表中"
            
            # 添加股票
            account.watch_list.append(symbol)
            
            if self._save_account(run_context, account):
                return f"成功添加股票 {symbol} 到自选股列表"
            else:
                return "错误：保存自选股列表失败"
            
        except Exception as e:
            return f"添加自选股时发生错误: {str(e)}"

    def remove_watch_stock(self, run_context: RunContext, symbol: str) -> str:
        """
        移除自选股
        
        Args:
            symbol: 股票代码
        """
        try:
            account = self._get_account(run_context)
            if not account or not account.watch_list:
                return "错误：未找到自选股列表"
            
            if symbol in account.watch_list:
                account.watch_list.remove(symbol)
                if self._save_account(run_context, account):
                    return f"成功从自选股列表中移除股票 {symbol}"
                else:
                    return "错误：保存自选股列表失败"
            else:
                return f"错误：股票 {symbol} 不在自选股列表中"
                
        except Exception as e:
            return f"移除自选股时发生错误: {str(e)}"

    def add_market_index(self, run_context: RunContext, index_code: str) -> str:
        """
        添加市场指数到关注列表
        
        Args:
            index_code: 指数代码
        """
        try:
            account = self._get_account(run_context)
            if not account:
                return "错误：未找到账户信息"
            
            # 初始化指数列表
            if not account.market_index_list:
                account.market_index_list = []
            
            # 检查是否已存在
            if index_code in account.market_index_list:
                return f"指数 {index_code} 已在关注列表中"
            
            # 添加指数
            account.market_index_list.append(index_code)
            
            if self._save_account(run_context, account):
                return f"成功添加指数 {index_code} 到关注列表"
            else:
                return "错误：保存指数列表失败"
            
        except Exception as e:
            return f"添加指数时发生错误: {str(e)}"

    def remove_market_index(self, run_context: RunContext, index_code: str) -> str:
        """
        移除市场指数关注
        
        Args:
            index_code: 指数代码
        """
        try:
            account = self._get_account(run_context)
            if not account or not account.market_index_list:
                return "错误：未找到指数关注列表"
            
            if index_code in account.market_index_list:
                account.market_index_list.remove(index_code)
                if self._save_account(run_context, account):
                    return f"成功从关注列表中移除指数 {index_code}"
                else:
                    return "错误：保存指数列表失败"
            else:
                return f"错误：指数 {index_code} 不在关注列表中"
                
        except Exception as e:
            return f"移除指数时发生错误: {str(e)}"