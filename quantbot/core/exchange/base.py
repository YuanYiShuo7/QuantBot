from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging
from pydantic import BaseModel, Field

from .order_types import Order, OrderInterface, OrderStatus, OrderType, OrderList
from .result_types import Result, ResultList
from .obervation_types import ObervationSpace, AccountData, PositionData, StockData

class MockExchange:
    """模拟交易所接口类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化模拟交易所"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 交易参数配置
        self.trading_fee_rate = self.config.get('trading_fee_rate', 0.001)  # 交易费率 0.1%
        self.stamp_tax_rate = self.config.get('stamp_tax_rate', 0.001)      # 印花税 0.1% (仅卖出时收取)
        self.min_trading_unit = self.config.get('min_trading_unit', 100)    # 最小交易单位
        
        # 交易规则
        self.price_limit_rate = self.config.get('price_limit_rate', 0.1)    # 涨跌幅限制 10%
        self.t_plus_one = self.config.get('t_plus_one', True)               # T+1交易制度
        
        self.logger.info("模拟交易所初始化完成")
    
    def update_order(self, order_interface: OrderInterface, observation: ObervationSpace, timestamp: datetime) -> ObervationSpace:
        """
        更新订单方法 - 将OrderInterface转换为完整Order并添加到observation
        
        Args:
            order_interface: 简化的订单接口对象
            observation: 当前市场观测状态
            timestamp: 时间戳
            
        Returns:
            更新后的observation
        """
        try:
            # 验证交易时间
            if not self._is_trading_time(timestamp):
                self.logger.warning(f"非交易时间: {timestamp}")
                return observation
            
            # 验证市场状态
            if not observation.trading_enabled:
                self.logger.warning("交易功能未启用")
                return observation
            
            # 验证订单基本规则
            validation_result = self._validate_order_interface(order_interface, observation)
            if not validation_result[0]:
                self.logger.warning(f"订单验证失败: {validation_result[1]}")
                return observation
            
            # 创建完整订单对象
            order = self._create_order_from_interface(order_interface, timestamp)
            
            # 添加到observation的订单列表
            observation.orders.orders.append(order)
            
            self.logger.info(f"订单更新成功: {order.order_id} - {order.symbol} {order.order_type} {order.quantity}@{order.price}")
            
            return observation
            
        except Exception as e:
            self.logger.error(f"更新订单失败: {e}")
            return observation
    
    def execute_orders(self, observation: ObervationSpace, timestamp: datetime) -> Tuple[ObervationSpace, ResultList]:
        """
        完成订单交易方法 - 检查并执行符合条件的订单
        
        Args:
            observation: 当前市场观测状态
            timestamp: 时间戳
            
        Returns:
            Tuple[更新后的observation, 交易结果列表]
        """
        results = ResultList()
        updated_observation = observation.copy()
        
        try:
            # 检查交易条件
            if not self._is_trading_time(timestamp):
                self.logger.warning(f"非交易时间，跳过订单执行: {timestamp}")
                return updated_observation, results
            
            if not observation.trading_enabled:
                self.logger.warning("交易功能未启用，跳过订单执行")
                return updated_observation, results
            
            # 处理待处理订单
            pending_orders = [order for order in observation.orders.orders if order.status == OrderStatus.PENDING]
            
            for order in pending_orders:
                result = self._try_execute_order(order, updated_observation, timestamp)
                if result:
                    results.results.append(result)
                    
                    # 更新订单状态
                    order.status = result.status
                    
                    # 如果交易成功，更新账户和持仓
                    if result.status == OrderStatus.SUCCESS:
                        updated_observation = self._update_account_and_positions(
                            updated_observation, order, result, timestamp
                        )
            
            self.logger.info(f"订单执行完成: 处理{len(pending_orders)}个订单，成交{len(results.results)}笔")
            
            return updated_observation, results
            
        except Exception as e:
            self.logger.error(f"执行订单失败: {e}")
            return updated_observation, results
    
    def _is_trading_time(self, timestamp: datetime) -> bool:
        """检查是否为交易时间"""
        # 简单的交易时间检查（实际应用中需要更复杂的逻辑）
        hour = timestamp.hour
        minute = timestamp.minute
        
        # 上午交易时间: 9:30-11:30
        morning_start = (9, 30)
        morning_end = (11, 30)
        
        # 下午交易时间: 13:00-15:00
        afternoon_start = (13, 0)
        afternoon_end = (15, 0)
        
        # 检查是否在上午交易时段
        if (hour > morning_start[0] or (hour == morning_start[0] and minute >= morning_start[1])):
            if (hour < morning_end[0] or (hour == morning_end[0] and minute <= morning_end[1])):
                return True
        
        # 检查是否在下午交易时段
        if (hour > afternoon_start[0] or (hour == afternoon_start[0] and minute >= afternoon_start[1])):
            if (hour < afternoon_end[0] or (hour == afternoon_end[0] and minute <= afternoon_end[1])):
                return True
        
        return False
    
    def _validate_order_interface(self, order_interface: OrderInterface, observation: ObervationSpace) -> Tuple[bool, str]:
        """验证订单接口的合法性"""
        symbol = order_interface.symbol
        
        # 检查股票是否存在
        if symbol not in observation.stock_data:
            return False, f"股票{symbol}不存在"
        
        # 获取当前股价
        current_price = observation.stock_data[symbol].real_time.current_price
        prev_close = observation.stock_data[symbol].real_time.pre_close
        
        # 检查价格限制
        price_limit = prev_close * self.price_limit_rate
        if order_interface.price > prev_close * (1 + self.price_limit_rate):
            return False, f"买入价格超过涨停限制 {prev_close * (1 + self.price_limit_rate):.2f}"
        if order_interface.price < prev_close * (1 - self.price_limit_rate):
            return False, f"卖出价格低于跌停限制 {prev_close * (1 - self.price_limit_rate):.2f}"
        
        # 检查最小交易单位
        if order_interface.quantity % self.min_trading_unit != 0:
            return False, f"数量必须是{self.min_trading_unit}的整数倍"
        
        # 检查资金或持仓
        if order_interface.order_type == OrderType.BUY:
            total_cost = order_interface.price * order_interface.quantity
            if total_cost > observation.account_info.available_cash:
                return False, f"资金不足，需要{total_cost:.2f}，可用{observation.account_info.available_cash:.2f}"
        else:  # SELL
            # 查找持仓
            position = next((p for p in observation.positions if p.symbol == symbol), None)
            if not position or position.available_quantity < order_interface.quantity:
                return False, f"持仓不足，需要卖出{order_interface.quantity}，可用{position.available_quantity if position else 0}"
        
        return True, "验证通过"
    
    def _create_order_from_interface(self, order_interface: OrderInterface, timestamp: datetime) -> Order:
        """从OrderInterface创建完整的Order对象"""
        return Order(
            order_id=f"order_{uuid.uuid4().hex[:8]}",
            symbol=order_interface.symbol,
            order_type=order_interface.order_type,
            price=order_interface.price,
            quantity=order_interface.quantity,
            timestamp=timestamp,
            status=OrderStatus.PENDING
        )
    
    def _try_execute_order(self, order: Order, observation: ObervationSpace, timestamp: datetime) -> Optional[Result]:
        """尝试执行单个订单"""
        symbol = order.symbol
        
        # 检查股票数据是否存在
        if symbol not in observation.stock_data:
            return Result.failure(
                content=f"股票{symbol}数据不存在",
                error_reason="股票数据缺失",
                order_id=order.order_id
            )
        
        # 获取当前市场数据
        stock_data = observation.stock_data[symbol]
        current_price = stock_data.real_time.current_price
        
        # 检查价格条件
        can_execute = False
        if order.order_type == OrderType.BUY:
            # 买入订单：当前价 <= 订单价 时可以执行
            can_execute = current_price <= order.price
        else:  # SELL
            # 卖出订单：当前价 >= 订单价 时可以执行
            can_execute = current_price >= order.price
        
        if not can_execute:
            return None
        
        # 计算交易费用
        transaction_amount = order.quantity * current_price
        trading_fee = transaction_amount * self.trading_fee_rate
        stamp_tax = transaction_amount * self.stamp_tax_rate if order.order_type == OrderType.SELL else 0
        total_fee = trading_fee + stamp_tax
        
        # 创建成功结果
        return Result.success(
            content=f"{order.order_type}订单执行成功",
            order_id=order.order_id,
            executed_price=current_price,
            executed_quantity=order.quantity,
            remaining_quantity=0,  # 假设全部成交
            fee=total_fee
        )
    
    def _update_account_and_positions(self, observation: ObervationSpace, order: Order, result: Result, timestamp: datetime) -> ObervationSpace:
        """更新账户和持仓信息"""
        symbol = order.symbol
        quantity = order.quantity
        price = result.executed_price
        total_amount = quantity * price
        fee = result.fee or 0
        
        # 更新账户信息
        if order.order_type == OrderType.BUY:
            # 买入：减少现金，增加持仓市值
            observation.account_info.available_cash -= (total_amount + fee)
            observation.account_info.market_value += total_amount
        else:  # SELL
            # 卖出：增加现金，减少持仓市值
            observation.account_info.available_cash += (total_amount - fee)
            observation.account_info.market_value -= total_amount
        
        # 更新总资产
        observation.account_info.total_assets = (
            observation.account_info.available_cash + observation.account_info.market_value
        )
        
        # 更新持仓比例
        if observation.account_info.total_assets > 0:
            observation.account_info.position_rate = (
                observation.account_info.market_value / observation.account_info.total_assets
            )
        
        # 更新时间戳
        observation.account_info.timestamp = timestamp
        observation.timestamp = timestamp
        
        # 更新持仓信息
        self._update_positions(observation, order, result, timestamp)
        
        return observation
    
    def _update_positions(self, observation: ObervationSpace, order: Order, result: Result, timestamp: datetime):
        """更新持仓信息"""
        symbol = order.symbol
        quantity = order.quantity
        price = result.executed_price
        
        # 查找现有持仓
        position_index = -1
        for i, pos in enumerate(observation.positions):
            if pos.symbol == symbol:
                position_index = i
                break
        
        if order.order_type == OrderType.BUY:
            if position_index >= 0:
                # 更新现有持仓
                position = observation.positions[position_index]
                total_cost = (position.cost_price * position.quantity) + (price * quantity)
                total_quantity = position.quantity + quantity
                
                position.quantity = total_quantity
                position.available_quantity = total_quantity  # 简化处理，实际T+1需要延迟
                position.cost_price = total_cost / total_quantity
                position.current_price = price
                position.market_value = total_quantity * price
                position.timestamp = timestamp
                
                # 重新计算盈亏
                position.profit_loss = position.market_value - total_cost
                position.profit_loss_rate = position.profit_loss / total_cost if total_cost > 0 else 0
            else:
                # 创建新持仓
                new_position = PositionData(
                    timestamp=timestamp,
                    symbol=symbol,
                    name=observation.stock_data[symbol].real_time.name,
                    quantity=quantity,
                    available_quantity=quantity,
                    cost_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    profit_loss=0,
                    profit_loss_rate=0
                )
                observation.positions.append(new_position)
        
        else:  # SELL
            if position_index >= 0:
                position = observation.positions[position_index]
                
                # 减少持仓数量
                position.quantity -= quantity
                position.available_quantity -= quantity
                position.market_value = position.quantity * price
                position.current_price = price
                position.timestamp = timestamp
                
                # 重新计算盈亏（基于剩余持仓）
                total_cost = position.cost_price * position.quantity
                position.profit_loss = position.market_value - total_cost
                position.profit_loss_rate = position.profit_loss / total_cost if total_cost > 0 else 0
                
                # 如果持仓为0，移除该持仓
                if position.quantity <= 0:
                    observation.positions.pop(position_index)
    
    def cancel_order(self, order_id: str, observation: ObervationSpace, timestamp: datetime) -> Tuple[ObervationSpace, Result]:
        """取消指定订单"""
        # 查找订单
        order_index = -1
        for i, order in enumerate(observation.orders.orders):
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order_index = i
                break
        
        if order_index >= 0:
            # 取消订单
            order = observation.orders.orders[order_index]
            order.status = OrderStatus.EXPIRED
            
            result = Result(
                order=order,
                status=OrderStatus.EXPIRED,
                content=f"订单{order_id}已取消",
                timestamp=timestamp
            )
            
            self.logger.info(f"订单取消成功: {order_id}")
            return observation, result
        else:
            # 订单不存在或无法取消
            result = Result.failure(
                content=f"订单{order_id}取消失败",
                error_reason="订单不存在或无法取消",
                order_id=order_id
            )
            
            self.logger.warning(f"订单取消失败: {order_id}")
            return observation, result
    
    def get_order_status(self, order_id: str, observation: ObervationSpace) -> Optional[Order]:
        """获取订单状态"""
        for order in observation.orders.orders:
            if order.order_id == order_id:
                return order
        return None