from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import uuid
from enum import Enum

from general.schemas.account_schema import AccountSchema, AccountData, PositionData
from general.schemas.order_schema import OrderSchema, OrderFormSchema, OrderResultSchema, OrderStatus, OrderType
from general.schemas.action_schema import ActionSchemaUnion, AddOrderActionSchema, CancelOrderActionSchema, NoneActionSchema, ActionType
from general.interfaces.exchange_interface import ExchangeInterface
class Exchange(ExchangeInterface):
    """默认模拟交易所实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 交易费用配置
        self.commission_rate = self.config.get('commission_rate', 0.0003)  # 佣金费率，默认万分之三
        self.min_commission = self.config.get('min_commission', 5.0)  # 最低佣金，默认5元
        self.stamp_duty_rate = self.config.get('stamp_duty_rate', 0.001)  # 印花税率，默认千分之一（仅卖出收取）
        self.transfer_fee_rate = self.config.get('transfer_fee_rate', 0.00002)  # 过户费率，默认万分之0.2
        
        # 订单簿
        self.order_book: Dict[str, OrderSchema] = {}
        
        # 订单执行配置
        self.execution_slippage = self.config.get('execution_slippage', 0.001)  # 执行滑点，默认0.1%
        self.min_execution_volume = self.config.get('min_execution_volume', 100)  # 最小执行量
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("模拟交易所初始化完成")
        self.logger.info(f"交易费用 - 佣金: {self.commission_rate*10000}‰, 印花税: {self.stamp_duty_rate*1000}‰, 过户费: {self.transfer_fee_rate*10000}‰")
    
    def update_orders(self, actions: List[ActionSchemaUnion], account: AccountSchema, timestamp: datetime) -> List[OrderSchema]:
        """根据动作更新订单"""
        new_orders = []
        
        for action in actions:
            try:
                if isinstance(action, AddOrderActionSchema):
                    # 创建新订单
                    order = self._create_order_from_action(action, timestamp)
                    
                    # 验证订单
                    validation_result = self._validate_order(order, account)
                    if not validation_result[0]:
                        self.logger.warning(f"订单验证失败: {validation_result[1]}")
                        continue
                    
                    # 添加到订单簿
                    self.order_book[order.order_id] = order
                    new_orders.append(order)
                    
                    self.logger.info(f"新增订单: {order.order_id}, {order.symbol}, {order.order_type}, {order.quantity}股 @ {order.price}")
                    
                elif isinstance(action, CancelOrderActionSchema):
                    # 取消订单
                    if action.order_id in self.order_book:
                        cancelled_order = self.order_book[action.order_id]
                        cancelled_order.status = OrderStatus.EXPIRED
                        del self.order_book[action.order_id]
                        self.logger.info(f"取消订单: {action.order_id}")
                    else:
                        self.logger.warning(f"取消订单失败: 订单 {action.order_id} 不存在")
                        
                elif isinstance(action, NoneActionSchema):
                    # 无操作
                    self.logger.debug(f"无操作: {action.reasoning}")
                    
            except Exception as e:
                self.logger.error(f"处理动作失败: {str(e)}")
                continue
        
        return new_orders
    
    def _create_order_from_action(self, action: AddOrderActionSchema, timestamp: datetime) -> OrderSchema:
        """从动作创建订单"""
        order_id = f"ORDER_{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        return OrderSchema(
            order_id=order_id,
            symbol=action.order_form.symbol,
            order_type=action.order_form.order_type,
            price=action.order_form.price,
            quantity=action.order_form.quantity,
            timestamp=timestamp,
            status=OrderStatus.PENDING
        )
    
    def _validate_order(self, order: OrderSchema, account: AccountSchema) -> Tuple[bool, str]:
        """验证订单是否有效"""
        try:
            # 检查订单数量（A股必须是100的整数倍）
            if order.quantity % 100 != 0:
                return False, "订单数量必须是100的整数倍"
            
            # 检查买入订单的资金是否充足
            if order.order_type == OrderType.BUY:
                total_cost = order.price * order.quantity
                required_funds = total_cost * (1 + self.commission_rate)  # 考虑佣金
                
                if required_funds > account.account_info.available_cash:
                    return False, f"资金不足，需要{required_funds:.2f}，可用{account.account_info.available_cash:.2f}"
            
            # 检查卖出订单的持仓是否充足
            elif order.order_type == OrderType.SELL:
                # 查找对应持仓
                position = None
                for pos in account.positions:
                    if pos.symbol == order.symbol:
                        position = pos
                        break
                
                if not position:
                    return False, f"没有 {order.symbol} 的持仓"
                
                if order.quantity > position.available_quantity:
                    return False, f"持仓不足，需要{order.quantity}，可用{position.available_quantity}"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证异常: {str(e)}"
    
    def check_and_execute_orders(self, market, account: AccountSchema, timestamp: datetime) -> List[OrderResultSchema]:
        """检查并执行订单"""
        order_results = []
        orders_to_remove = []
        
        for order_id, order in self.order_book.items():
            try:
                # 检查订单是否过期（超过当日）
                if order.timestamp.date() < timestamp.date():
                    order.status = OrderStatus.EXPIRED
                    orders_to_remove.append(order_id)
                    self.logger.info(f"订单过期: {order_id}")
                    continue
                
                # 获取当前市场数据
                if order.symbol not in market.stock_data:
                    self.logger.warning(f"无法执行订单 {order_id}: 股票 {order.symbol} 不在市场数据中")
                    continue
                
                current_price = market.stock_data[order.symbol].real_time.current_price
                
                # 检查订单是否可执行
                execution_result = self._check_order_execution(order, current_price, timestamp)
                
                if execution_result[0]:
                    # 执行订单
                    order_result = self._execute_order(order, current_price, account, timestamp)
                    order_results.append(order_result)
                    
                    # 如果订单完全成交或失败，从订单簿移除
                    if order_result.status in [OrderStatus.SUCCESS, OrderStatus.FAILED]:
                        orders_to_remove.append(order_id)
                    
                    self.logger.info(f"订单执行: {order_id}, 状态: {order_result.status}")
                
            except Exception as e:
                self.logger.error(f"检查订单 {order_id} 失败: {str(e)}")
                continue
        
        # 移除已处理订单
        for order_id in orders_to_remove:
            if order_id in self.order_book:
                del self.order_book[order_id]
        
        return order_results
    
    def _check_order_execution(self, order: OrderSchema, current_price: float, timestamp: datetime) -> Tuple[bool, str]:
        """检查订单是否可执行"""
        try:
            # 买入订单：当前价格 <= 订单价格
            if order.order_type == OrderType.BUY:
                if current_price <= order.price:
                    return True, f"买入条件满足: 当前价{current_price} <= 订单价{order.price}"
                else:
                    return False, f"买入条件不满足: 当前价{current_price} > 订单价{order.price}"
            
            # 卖出订单：当前价格 >= 订单价格
            elif order.order_type == OrderType.SELL:
                if current_price >= order.price:
                    return True, f"卖出条件满足: 当前价{current_price} >= 订单价{order.price}"
                else:
                    return False, f"卖出条件不满足: 当前价{current_price} < 订单价{order.price}"
            
            return False, "未知订单类型"
            
        except Exception as e:
            return False, f"检查执行条件失败: {str(e)}"
    
    def _execute_order(self, order: OrderSchema, execution_price: float, account: AccountSchema, timestamp: datetime) -> OrderResultSchema:
        """执行订单"""
        try:
            # 应用滑点
            if order.order_type == OrderType.BUY:
                # 买入时价格可能更高
                actual_price = execution_price * (1 + self.execution_slippage)
            else:
                # 卖出时价格可能更低
                actual_price = execution_price * (1 - self.execution_slippage)
            
            actual_price = round(actual_price, 2)  # A股价格精度为2位小数
            
            # 计算成交金额
            executed_amount = actual_price * order.quantity
            
            # 计算交易费用
            commission, stamp_duty, transfer_fee = self._calculate_fees(order, executed_amount)
            total_fee = commission + stamp_duty + transfer_fee
            
            # 计算净金额
            if order.order_type == OrderType.BUY:
                net_amount = executed_amount + total_fee
            else:
                net_amount = executed_amount - total_fee
            
            # 创建订单结果
            order_result = OrderResultSchema(
                order_id=order.order_id,
                symbol=order.symbol,
                order_type=order.order_type,
                status=OrderStatus.SUCCESS,
                timestamp=timestamp,
                executed_quantity=order.quantity,
                executed_price=actual_price,
                executed_amount=executed_amount,
                commission=commission,
                stamp_duty=stamp_duty,
                transfer_fee=transfer_fee,
                total_fee=total_fee,
                net_amount=net_amount,
                error_message=None,
                error_code=None
            )
            
            # 更新账户
            self._update_account_with_order(account, order_result)
            
            # 更新订单状态
            order.status = OrderStatus.SUCCESS
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"执行订单 {order.order_id} 失败: {str(e)}")
            
            # 创建失败结果
            return OrderResultSchema(
                order_id=order.order_id,
                symbol=order.symbol,
                order_type=order.order_type,
                status=OrderStatus.FAILED,
                timestamp=timestamp,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    def _calculate_fees(self, order: OrderSchema, executed_amount: float) -> Tuple[float, float, float]:
        """计算交易费用"""
        # 佣金（双向收取）
        commission = max(executed_amount * self.commission_rate, self.min_commission)
        
        # 印花税（仅卖出收取）
        stamp_duty = executed_amount * self.stamp_duty_rate if order.order_type == OrderType.SELL else 0.0
        
        # 过户费（双向收取，仅沪市股票）
        transfer_fee = 0.0
        if order.symbol.startswith('6'):  # 沪市股票
            transfer_fee = executed_amount * self.transfer_fee_rate
        
        return commission, stamp_duty, transfer_fee
    
    def _update_account_with_order(self, account: AccountSchema, order_result: OrderResultSchema) -> None:
        """根据订单结果更新账户"""
        try:
            # 更新账户资金
            if order_result.order_type == OrderType.BUY:
                account.account_info.available_cash -= order_result.net_amount
            else:
                account.account_info.available_cash += order_result.net_amount
            
            # 更新持仓
            self._update_positions(account, order_result)
            
            # 重新计算账户统计
            self._recalculate_account_stats(account)
            
            # 添加订单到账户历史
            executed_order = OrderSchema(
                order_id=order_result.order_id,
                symbol=order_result.symbol,
                order_type=order_result.order_type,
                price=order_result.executed_price,
                quantity=order_result.executed_quantity,
                timestamp=order_result.timestamp,
                status=order_result.status
            )
            account.orders.append(executed_order)
            
        except Exception as e:
            self.logger.error(f"更新账户失败: {str(e)}")
            raise
    
    def _update_positions(self, account: AccountSchema, order_result: OrderResultSchema) -> None:
        """更新持仓信息"""
        symbol = order_result.symbol
        
        if order_result.order_type == OrderType.BUY:
            # 买入操作
            position = self._find_or_create_position(account, symbol, "未知股票")
            
            # 计算新的平均成本
            total_quantity = position.quantity + order_result.executed_quantity
            total_cost = (position.cost_price * position.quantity + 
                         order_result.executed_amount)
            
            position.cost_price = total_cost / total_quantity
            position.quantity = total_quantity
            position.available_quantity = total_quantity  # 假设T+1，当日买入下一日才可卖
            
        else:
            # 卖出操作
            position = self._find_position(account, symbol)
            if position:
                position.quantity -= order_result.executed_quantity
                position.available_quantity -= order_result.executed_quantity
                
                # 如果持仓为0，移除该持仓
                if position.quantity <= 0:
                    account.positions = [p for p in account.positions if p.symbol != symbol]
    
    def _find_or_create_position(self, account: AccountSchema, symbol: str, name: str) -> PositionData:
        """查找或创建持仓"""
        position = self._find_position(account, symbol)
        if not position:
            position = PositionData(
                timestamp=datetime.now(),
                symbol=symbol,
                name=name,
                quantity=0,
                available_quantity=0,
                cost_price=0.0,
                current_price=0.0,
                market_value=0.0,
                profit_loss=0.0,
                profit_loss_rate=0.0
            )
            account.positions.append(position)
        return position
    
    def _find_position(self, account: AccountSchema, symbol: str) -> Optional[PositionData]:
        """查找持仓"""
        for position in account.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _recalculate_account_stats(self, account: AccountSchema) -> None:
        """重新计算账户统计信息"""
        try:
            # 计算持仓市值和盈亏
            total_market_value = 0.0
            total_profit_loss = 0.0
            
            for position in account.positions:
                # 注意：这里需要从市场数据获取当前价格，暂时使用成本价
                # 在实际使用中，应该通过market接口获取实时价格
                position.current_price = position.cost_price  # 应该从市场接口获取
                position.market_value = position.quantity * position.current_price
                position.profit_loss = (position.current_price - position.cost_price) * position.quantity
                position.profit_loss_rate = (position.current_price / position.cost_price - 1) * 100 if position.cost_price > 0 else 0
                
                total_market_value += position.market_value
                total_profit_loss += position.profit_loss
            
            # 更新账户信息
            account.account_info.market_value = total_market_value
            account.account_info.total_assets = account.account_info.available_cash + total_market_value
            account.account_info.total_profit_loss = total_profit_loss
            account.account_info.total_profit_loss_rate = (total_profit_loss / (account.account_info.total_assets - total_profit_loss)) * 100 if (account.account_info.total_assets - total_profit_loss) > 0 else 0
            account.account_info.position_rate = (total_market_value / account.account_info.total_assets) * 100 if account.account_info.total_assets > 0 else 0
            
        except Exception as e:
            self.logger.error(f"重新计算账户统计失败: {str(e)}")