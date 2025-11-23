from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libs.agno.tools import Toolkit
from libs.agno.run import RunContext

from general.schemas.order_schema import OrderSchema, OrderStatus, OrderType, OrderFormSchema
from general.schemas.account_schema import AccountSchema, AccountData, PositionData

class OrderTools(Toolkit):
    """订单工具类"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="order_tools", 
            tools=[
                self.create_order,
                self.cancel_order,
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

    def _update_account_after_order(self, account: AccountSchema, order: OrderSchema) -> None:
        """创建订单后更新账户元信息"""
        try:
            # 更新账户时间戳
            account.timestamp = datetime.now()
            account.account_info.timestamp = datetime.now()
            
            if order.order_type == OrderType.BUY:
                # 买入订单：冻结资金
                order_amount = order.price * order.quantity
                account.account_info.available_cash -= order_amount
                
                # 更新总资产（资金减少，但持仓价值还未增加）
                account.account_info.total_assets = (
                    account.account_info.available_cash + 
                    account.account_info.market_value
                )
                
            elif order.order_type == OrderType.SELL:
                # 卖出订单：冻结持仓
                position = self._find_position(account, order.symbol)
                if position:
                    position.available_quantity -= order.quantity
            
            # 重新计算持仓比例
            self._update_position_rate(account)
            
        except Exception as e:
            print(f"更新账户元信息时发生错误: {str(e)}")

    def _update_account_after_cancel(self, account: AccountSchema, order: OrderSchema) -> None:
        """取消订单后更新账户元信息"""
        try:
            # 更新账户时间戳
            account.timestamp = datetime.now()
            account.account_info.timestamp = datetime.now()
            
            if order.order_type == OrderType.BUY:
                # 取消买入订单：释放冻结资金
                order_amount = order.price * order.quantity
                account.account_info.available_cash += order_amount
                
            elif order.order_type == OrderType.SELL:
                # 取消卖出订单：释放冻结持仓
                position = self._find_position(account, order.symbol)
                if position:
                    position.available_quantity += order.quantity
            
            # 重新计算总资产和持仓比例
            account.account_info.total_assets = (
                account.account_info.available_cash + 
                account.account_info.market_value
            )
            self._update_position_rate(account)
            
        except Exception as e:
            print(f"取消订单后更新账户元信息时发生错误: {str(e)}")

    def _find_position(self, account: AccountSchema, symbol: str) -> Optional[PositionData]:
        """查找持仓"""
        for position in account.positions:
            if position.symbol == symbol:
                return position
        return None

    def _update_position_rate(self, account: AccountSchema) -> None:
        """更新持仓比例"""
        if account.account_info.total_assets > 0:
            account.account_info.position_rate = (
                account.account_info.market_value / account.account_info.total_assets * 100
            )
        else:
            account.account_info.position_rate = 0.0

    def _validate_buy_order(self, account: AccountSchema, price: float, quantity: float) -> str:
        """验证买入订单的可行性"""
        required_amount = price * quantity
        if account.account_info.available_cash < required_amount:
            return f"错误：可用资金不足。需要{required_amount:.2f}元，当前可用{account.account_info.available_cash:.2f}元"
        return ""

    def _validate_sell_order(self, account: AccountSchema, symbol: str, quantity: float) -> str:
        """验证卖出订单的可行性"""
        position = self._find_position(account, symbol)
        if not position:
            return f"错误：未持有股票{symbol}"
        
        if position.available_quantity < quantity:
            return f"错误：可用持仓不足。需要{quantity}股，当前可用{position.available_quantity}股"
        return ""

    def create_order(
        self, 
        run_context: RunContext,
        symbol: str,
        order_type: str,
        price: float,
        quantity: float
    ) -> str:
        """
        创建新订单
        
        Args:
            symbol: 股票代码，6位数字
            order_type: 订单类型，BUY/SELL
            price: 价格，大于0
            quantity: 数量，必须为100的整数倍
        """
        try:
            # 验证参数
            if not symbol or len(symbol) != 6 or not symbol.isdigit():
                return "错误：股票代码必须为6位数字"
            
            if order_type.upper() not in ["BUY", "SELL"]:
                return "错误：订单类型必须是BUY或SELL"
            
            if price <= 0:
                return "错误：价格必须大于0"
            
            if quantity <= 0 or quantity % 100 != 0:
                return "错误：数量必须大于0且为100的整数倍"
            
            # 获取账户状态
            account = self._get_account(run_context)
            if not account:
                return "错误：未找到账户信息"
            
            # 验证订单可行性
            order_type_enum = OrderType(order_type.upper())
            if order_type_enum == OrderType.BUY:
                validation_result = self._validate_buy_order(account, price, quantity)
                if validation_result:
                    return validation_result
            else:  # SELL
                validation_result = self._validate_sell_order(account, symbol, quantity)
                if validation_result:
                    return validation_result
            
            # 创建订单ID
            order_id = f"order_{int(datetime.now().timestamp() * 1000)}"
            
            # 创建新订单
            new_order = OrderSchema(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type_enum,
                price=round(price, 2),
                quantity=quantity,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING
            )
            
            # 更新账户中的订单列表
            if not account.orders:
                account.orders = []
            
            account.orders.append(new_order)
            
            # 更新账户元信息（冻结资金或持仓）
            self._update_account_after_order(account, new_order)
            
            # 保存更新后的账户
            if self._save_account(run_context, account):
                order_type_text = "买入" if order_type_enum == OrderType.BUY else "卖出"
                return (f"订单创建成功！\n"
                       f"订单ID: {order_id}\n"
                       f"股票: {symbol}\n"
                       f"类型: {order_type_text}\n"
                       f"价格: {price:.2f}元\n"
                       f"数量: {quantity}股\n"
                       f"金额: {price * quantity:.2f}元\n"
                       f"当前可用资金: {account.account_info.available_cash:.2f}元")
            else:
                return "错误：保存订单失败"
            
        except Exception as e:
            return f"创建订单时发生错误: {str(e)}"

    def cancel_order(self, run_context: RunContext, order_id: str) -> str:
        """
        取消订单
        
        Args:
            order_id: 订单ID
        """
        try:
            account = self._get_account(run_context)
            if not account or not account.orders:
                return "错误：未找到订单列表"
            
            # 查找订单
            order_to_cancel = None
            for order in account.orders:
                if order.order_id == order_id:
                    order_to_cancel = order
                    break
            
            if not order_to_cancel:
                return f"错误：未找到订单 {order_id}"
            
            if order_to_cancel.status != OrderStatus.PENDING:
                return f"错误：订单 {order_id} 状态为 {order_to_cancel.status}，无法取消"
            
            # 更新订单状态
            order_to_cancel.status = OrderStatus.FAILED
            
            # 更新账户元信息（释放冻结的资金或持仓）
            self._update_account_after_cancel(account, order_to_cancel)
            
            if self._save_account(run_context, account):
                order_type_text = "买入" if order_to_cancel.order_type == OrderType.BUY else "卖出"
                return (f"订单取消成功！\n"
                       f"订单ID: {order_id}\n"
                       f"股票: {order_to_cancel.symbol}\n"
                       f"类型: {order_type_text}\n"
                       f"当前可用资金: {account.account_info.available_cash:.2f}元")
            else:
                return "错误：保存订单状态失败"
                
        except Exception as e:
            return f"取消订单时发生错误: {str(e)}"