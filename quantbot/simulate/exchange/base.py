from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import logging

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from general.schemas.account_schema import AccountSchema, AccountData, PositionData
from general.schemas.order_schema import OrderFormSchema, OrderResultSchema, OrderSchema, OrderStatus, OrderType
from general.schemas.action_schema import ActionSchemaUnion, AddOrderActionSchema, CancelOrderActionSchema, NoneActionSchema, ActionType
from general.schemas.market_schema import MarketSchema, StockRealTimeData
from general.interfaces.exchange_interface import ExchangeInterface
from account.base import Account
from market.base import Market

class Exchange(ExchangeInterface):
    """模拟交易所实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 交易配置
        self.commission_rate = self.config.get('commission_rate', 0.0003)  # 佣金费率，默认万三
        self.stamp_duty_rate = self.config.get('stamp_duty_rate', 0.001)   # 印花税率，默认千一（仅卖出收取）
        self.transfer_fee_rate = self.config.get('transfer_fee_rate', 0.00002)  # 过户费率
        
        # 订单管理
        self.order_expiry_days = self.config.get('order_expiry_days', 7)    # 订单过期天数，默认7天
        self.order_cleanup_days = self.config.get('order_cleanup_days', 14)  # 订单清理天数，默认14天
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("模拟交易所初始化完成")
        self.logger.info(f"佣金费率: {self.commission_rate}")
        self.logger.info(f"印花税率: {self.stamp_duty_rate}")
        self.logger.info(f"订单过期天数: {self.order_expiry_days}")
    
    def update_orders(self, actions: List[ActionSchemaUnion], account: Account, timestamp: datetime):
        """
        根据操作更新订单列表和账户状态
        
        Args:
            actions: 操作列表
            account: 账户数据
            timestamp: 当前时间戳
        """
        try:
            self.logger.info(f"开始处理 {len(actions)} 个操作")
            
            # 获取账户schema
            account_schema = account.get_account_schema()
            
            for action in actions:
                if isinstance(action, AddOrderActionSchema):
                    self._process_add_order(action, account_schema, timestamp)
                elif isinstance(action, CancelOrderActionSchema):
                    self._process_cancel_order(action, account_schema, timestamp)
                elif isinstance(action, NoneActionSchema):
                    self.logger.info("无操作指令")
                else:
                    self.logger.warning(f"未知操作类型: {type(action)}")
            
            # 更新账户时间戳
            account_schema.timestamp = timestamp
            account_schema.account_info.timestamp = timestamp
            
            # 设置更新后的账户schema
            account.set_account_schema(account_schema)
            
            self.logger.info("操作处理完成")
            
        except Exception as e:
            self.logger.error(f"更新订单失败: {str(e)}")
            raise
    
    def _process_add_order(self, action: AddOrderActionSchema, account_schema: AccountSchema, timestamp: datetime):
        """处理添加订单操作"""
        try:
            order_form = action.order_form
            self.logger.info(f"处理添加订单: {order_form.symbol} {order_form.order_type} {order_form.quantity}@{order_form.price}")
            
            # 验证订单基本参数
            if order_form.quantity <= 0:
                raise ValueError("订单数量必须大于0")
            
            if order_form.price <= 0:
                raise ValueError("订单价格必须大于0")
            
            # 生成订单ID
            order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
            
            # 创建订单对象
            order = OrderSchema(
                order_id=order_id,
                symbol=order_form.symbol,
                order_type=order_form.order_type,
                price=order_form.price,
                quantity=order_form.quantity,
                timestamp=timestamp,
                status=OrderStatus.PENDING
            )
            
            # 检查资金或持仓是否足够
            if order.order_type == OrderType.BUY:
                # 买入订单：检查资金是否足够
                total_cost = order.price * order.quantity
                commission = total_cost * self.commission_rate
                total_amount = total_cost + commission
                
                if account_schema.account_info.available_cash < total_amount:
                    raise ValueError(f"资金不足，需要{total_amount:.2f}，可用{account_schema.account_info.available_cash:.2f}")
                
                # 冻结资金（在模拟中我们只是检查，不实际冻结）
                self.logger.info(f"买入订单资金检查通过: 需要{total_amount:.2f}，可用{account_schema.account_info.available_cash:.2f}")
                
            else:  # SELL
                # 卖出订单：检查持仓是否足够
                position = self._get_position(account_schema, order.symbol)
                if not position or position.available_quantity < order.quantity:
                    available_qty = position.available_quantity if position else 0
                    raise ValueError(f"持仓不足，需要{order.quantity}，可用{available_qty}")
                
                self.logger.info(f"卖出订单持仓检查通过: 需要{order.quantity}，可用{available_qty}")
            
            # 添加订单到账户
            account_schema.orders.append(order)
            self.logger.info(f"订单添加成功: {order_id}")
            
        except Exception as e:
            self.logger.error(f"添加订单失败: {str(e)}")
            raise
    
    def _process_cancel_order(self, action: CancelOrderActionSchema, account_schema: AccountSchema, timestamp: datetime):
        """处理取消订单操作"""
        try:
            order_id = action.order_id
            self.logger.info(f"处理取消订单: {order_id}")
            
            # 查找订单
            order_index = None
            for i, order in enumerate(account_schema.orders):
                if order.order_id == order_id:
                    order_index = i
                    break
            
            if order_index is None:
                raise ValueError(f"订单不存在: {order_id}")
            
            order = account_schema.orders[order_index]
            
            # 只能取消待处理的订单
            if order.status != OrderStatus.PENDING:
                raise ValueError(f"只能取消待处理订单，当前状态: {order.status}")
            
            # 移除订单
            account_schema.orders.pop(order_index)
            self.logger.info(f"订单取消成功: {order_id}")
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")
            raise
    
    def check_and_execute_orders(self, market: Market, account: Account, timestamp: datetime) -> List[OrderResultSchema]:
        """
        检查并执行订单，更新账户状态
        
        Args:
            market: 市场数据
            account: 账户数据
            timestamp: 当前时间戳
            
        Returns:
            订单执行结果列表
        """
        try:
            self.logger.info("开始检查并执行订单")
            executed_results = []
            
            # 获取账户和市场schema
            account_schema = account.get_account_schema()
            market_schema = market.get_market_schema()
            
            # 清理过期订单（超过两周）
            self._cleanup_expired_orders(account_schema, timestamp)
            
            # 检查每个待处理订单
            pending_orders = [order for order in account_schema.orders if order.status == OrderStatus.PENDING]
            self.logger.info(f"发现 {len(pending_orders)} 个待处理订单")
            
            for order in pending_orders:
                # 检查订单是否过期（超过一周）
                if self._is_order_expired(order, timestamp):
                    order.status = OrderStatus.EXPIRED
                    self.logger.info(f"订单已过期: {order.order_id}")
                    continue
                
                # 尝试执行订单
                result = self._try_execute_order(order, market_schema, account_schema, timestamp)
                if result:
                    executed_results.append(result)
                    self.logger.info(f"订单执行成功: {order.order_id}")
            
            # 更新账户资产数据
            self._update_account_assets(account_schema, market_schema, timestamp)
            
            # 设置更新后的账户schema
            account.set_account_schema(account_schema)
            
            self.logger.info(f"订单执行完成，共执行 {len(executed_results)} 个订单")
            return executed_results
            
        except Exception as e:
            self.logger.error(f"检查执行订单失败: {str(e)}")
            return []
    
    def _cleanup_expired_orders(self, account_schema: AccountSchema, timestamp: datetime):
        """清理过期订单（超过两周）"""
        cleanup_threshold = timestamp - timedelta(days=self.order_cleanup_days)
        account_schema.orders = [
            order for order in account_schema.orders 
            if order.timestamp > cleanup_threshold or order.status == OrderStatus.PENDING
        ]
    
    def _is_order_expired(self, order: OrderSchema, timestamp: datetime) -> bool:
        """检查订单是否过期（超过一周）"""
        expiry_threshold = timestamp - timedelta(days=self.order_expiry_days)
        return order.timestamp < expiry_threshold
    
    def _try_execute_order(self, order: OrderSchema, market_schema: MarketSchema, account_schema: AccountSchema, timestamp: datetime) -> Optional[OrderResultSchema]:
        """尝试执行单个订单"""
        try:
            # 获取股票实时数据
            stock_data = market_schema.stock_data.get(order.symbol)
            if not stock_data:
                self.logger.warning(f"股票数据不存在: {order.symbol}")
                return None
            
            real_time_data = stock_data.real_time
            
            # 检查价格是否在当日波动范围内
            if not (real_time_data.low <= order.price <= real_time_data.high):
                self.logger.debug(f"订单价格不在波动范围内: {order.price} not in [{real_time_data.low}, {real_time_data.high}]")
                return None
            
            # 检查订单执行条件
            if order.order_type == OrderType.BUY:
                return self._execute_buy_order(order, real_time_data, account_schema, timestamp)
            else:  # SELL
                return self._execute_sell_order(order, real_time_data, account_schema, timestamp)
                
        except Exception as e:
            self.logger.error(f"执行订单失败 {order.order_id}: {str(e)}")
            return None
    
    def _execute_buy_order(self, order: OrderSchema, real_time_data: StockRealTimeData, account_schema: AccountSchema, timestamp: datetime) -> Optional[OrderResultSchema]:
        """执行买入订单"""
        try:
            total_cost = order.price * order.quantity
            commission = total_cost * self.commission_rate
            transfer_fee = total_cost * self.transfer_fee_rate
            total_amount = total_cost + commission + transfer_fee
            
            # 检查资金是否足够
            if account_schema.account_info.available_cash < total_amount:
                self.logger.warning(f"资金不足，无法执行买入订单: 需要{total_amount:.2f}，可用{account_schema.account_info.available_cash:.2f}")
                return None
            
            # 扣除资金
            account_schema.account_info.available_cash -= total_amount
            
            # 更新或创建持仓
            position = self._get_position(account_schema, order.symbol)
            if position:
                # 更新现有持仓（成本价按加权平均计算）
                old_value = position.cost_price * position.quantity
                new_value = order.price * order.quantity
                total_quantity = position.quantity + order.quantity
                position.cost_price = (old_value + new_value) / total_quantity
                position.quantity = total_quantity
                position.available_quantity = total_quantity  # A股T+1，第二天才可用
            else:
                # 创建新持仓
                position = PositionData(
                    timestamp=timestamp,
                    symbol=order.symbol,
                    name=real_time_data.name,
                    quantity=order.quantity,
                    available_quantity=0,  # A股T+1，当天买入的第二天才可用
                    cost_price=order.price,
                    current_price=real_time_data.current_price,
                    market_value=order.quantity * real_time_data.current_price,
                    profit_loss=0.0,
                    profit_loss_rate=0.0
                )
                account_schema.positions.append(position)
            
            # 更新订单状态
            order.status = OrderStatus.SUCCESS
            
            # 创建执行结果
            return OrderResultSchema(
                order_id=order.order_id,
                symbol=order.symbol,
                order_type=order.order_type,
                status=OrderStatus.SUCCESS,
                timestamp=timestamp,
                executed_quantity=order.quantity,
                executed_price=order.price,
                executed_amount=total_cost,
                commission=commission
            )
            
        except Exception as e:
            self.logger.error(f"执行买入订单失败: {str(e)}")
            return None
    
    def _execute_sell_order(self, order: OrderSchema, real_time_data: StockRealTimeData, account_schema: AccountSchema, timestamp: datetime) -> Optional[OrderResultSchema]:
        """执行卖出订单"""
        try:
            # 检查持仓是否足够
            position = self._get_position(account_schema, order.symbol)
            if not position or position.available_quantity < order.quantity:
                available_qty = position.available_quantity if position else 0
                self.logger.warning(f"持仓不足，无法执行卖出订单: 需要{order.quantity}，可用{available_qty}")
                return None
            
            # 计算交易金额和费用
            total_amount = order.price * order.quantity
            commission = total_amount * self.commission_rate
            stamp_duty = total_amount * self.stamp_duty_rate  # 卖出收取印花税
            transfer_fee = total_amount * self.transfer_fee_rate
            net_amount = total_amount - commission - stamp_duty - transfer_fee
            
            # 更新持仓
            position.quantity -= order.quantity
            position.available_quantity -= order.quantity
            
            # 如果持仓为0，移除该持仓
            if position.quantity <= 0:
                account_schema.positions.remove(position)
            
            # 增加资金
            account_schema.account_info.available_cash += net_amount
            
            # 更新订单状态
            order.status = OrderStatus.SUCCESS
            
            # 创建执行结果
            return OrderResultSchema(
                order_id=order.order_id,
                symbol=order.symbol,
                order_type=order.order_type,
                status=OrderStatus.SUCCESS,
                timestamp=timestamp,
                executed_quantity=order.quantity,
                executed_price=order.price,
                executed_amount=total_amount,
                commission=commission
            )
            
        except Exception as e:
            self.logger.error(f"执行卖出订单失败: {str(e)}")
            return None
    
    def _get_position(self, account_schema: AccountSchema, symbol: str) -> Optional[PositionData]:
        """获取指定股票的持仓"""
        for position in account_schema.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _update_account_assets(self, account_schema: AccountSchema, market_schema: MarketSchema, timestamp: datetime):
        """更新账户资产数据"""
        try:
            # 更新持仓的当前价格和市值
            total_market_value = 0.0
            total_profit_loss = 0.0
            total_cost = 0.0
            
            for position in account_schema.positions:
                # 更新持仓的当前价格（从市场数据获取）
                stock_data = market_schema.stock_data.get(position.symbol)
                if stock_data:
                    current_price = stock_data.real_time.current_price
                    position.current_price = current_price
                    position.market_value = current_price * position.quantity
                    position.profit_loss = (current_price - position.cost_price) * position.quantity
                    position.profit_loss_rate = (current_price - position.cost_price) / position.cost_price * 100
                    
                    total_market_value += position.market_value
                    total_profit_loss += position.profit_loss
                    total_cost += position.cost_price * position.quantity
                
                # 更新持仓时间戳
                position.timestamp = timestamp
            
            # 更新账户信息
            account_info = account_schema.account_info
            account_info.timestamp = timestamp
            account_info.market_value = total_market_value
            account_info.total_assets = account_info.available_cash + total_market_value
            account_info.net_assets = account_info.total_assets
            account_info.total_profit_loss = total_profit_loss
            account_info.total_profit_loss_rate = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0.0
            account_info.position_rate = (total_market_value / account_info.total_assets * 100) if account_info.total_assets > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"更新账户资产数据失败: {str(e)}")