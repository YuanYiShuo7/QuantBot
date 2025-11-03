from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from general.interfaces.observation_interface import ObservationInterface
from general.types.obervation_types import ObervationSpace
from general.types.obervation_types import PositionData
from general.types.obervation_types import AccountData
from general.types.result_types import ResultList
from general.types.order_types import OrderList, Order, OrderStatus, OrderType
from config.observation_config import ObservationConfig
logger = logging.getLogger(__name__)

class DefaultObservation(ObservationInterface):
    """默认交易状态观察实现"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化接口
        Args:
            config: 配置参数字典
        """
        self.config = ObservationConfig(**(config or {}))
        self._current_observation: Optional[ObervationSpace] = None
        self._order_history: List[Order] = []  # 历史订单记录
        
        logger.info("DefaultObservation initialized with config: %s", self.config.dict())
    
    def initialize_observation(self, timestamp: datetime, market_data: Dict[str, Any]) -> ObervationSpace:
        """初始化交易状态
        
        Args:
            timestamp: 时间戳
            market_data: 市场数据
            
        Returns:
            ObervationSpace: 初始化的交易状态对象
        """
        # 创建初始账户信息
        account_info = self._create_initial_account(timestamp)
        
        # 创建初始持仓
        positions = self._create_initial_positions(timestamp)
        
        # 初始化市场数据
        market_indices, stock_data = self._initialize_market_data(market_data)
        
        # 创建初始观察空间
        self._current_observation = ObervationSpace(
            timestamp=timestamp,
            market_indices=market_indices,
            stock_data=stock_data,
            account_info=account_info,
            positions=positions,
            orders=OrderList(),
            watch_list=self.config.watch_list,
            market_index_list=self.config.market_index_list,
            trading_enabled=self.config.trading_enabled,
            market_status=self.config.market_status
        )
        
        logger.info("Observation initialized at %s", timestamp)
        return self._current_observation
    
    def update_obervation(self, timestamp: datetime, market_data: Dict[str, Any], 
                         order_results: ResultList) -> ObervationSpace:
        """更新交易状态
        
        Args:
            timestamp: 时间戳
            market_data: 最新市场数据
            order_results: 订单执行结果列表
            
        Returns:
            ObervationSpace: 更新后的交易状态
        """
        if self._current_observation is None:
            raise ValueError("Observation not initialized. Call initialize_observation first.")
        
        # 1. 获取当前订单列表并更新订单状态
        current_orders = self._current_observation.orders
        updated_orders = self._update_orders_from_results(current_orders, order_results, timestamp)
        
        # 2. 处理订单过期和清理
        processed_orders = self._process_order_expiry_and_cleanup(updated_orders, timestamp)
        
        # 3. 更新市场数据
        updated_market_indices, updated_stock_data = self._update_market_data(market_data)
        
        # 4. 更新账户和持仓信息（基于处理后的订单结果）
        updated_account = self._update_account_info(timestamp, order_results, processed_orders)
        updated_positions = self._update_positions(timestamp, order_results, processed_orders)
        
        # 5. 创建更新后的观察空间
        self._current_observation = ObervationSpace(
            timestamp=timestamp,
            market_indices=updated_market_indices,
            stock_data=updated_stock_data,
            account_info=updated_account,
            positions=updated_positions,
            orders=processed_orders,
            watch_list=self.config.watch_list,
            market_index_list=self.config.market_index_list,
            trading_enabled=self.config.trading_enabled,
            market_status=self._get_market_status(timestamp)
        )
        
        logger.debug("Observation updated at %s with %d order results processed", 
                    timestamp, len(order_results.results))
        return self._current_observation
    
    def get_obervation(self) -> ObervationSpace:
        """获取当前交易状态
        
        Returns:
            ObervationSpace: 当前交易状态
        """
        if self._current_observation is None:
            raise ValueError("Observation not initialized. Call initialize_observation first.")
        
        return self._current_observation
    
    def _update_orders_from_results(self, orders: OrderList, order_results: ResultList, 
                                   timestamp: datetime) -> OrderList:
        """根据交易结果更新订单状态
        
        Args:
            orders: 当前订单列表
            order_results: 交易结果列表
            timestamp: 当前时间戳
            
        Returns:
            OrderList: 更新后的订单列表
        """
        # 创建订单ID到订单的映射，便于快速查找
        order_map = {order.order_id: order for order in orders.orders}
        updated_orders = []
        processed_order_ids = set()
        
        success_count = 0
        failure_count = 0
        
        # 处理每个交易结果
        for result in order_results.results:
            order_id = result.order.order_id
            
            if order_id in order_map:
                # 更新现有订单
                existing_order = order_map[order_id]
                existing_order.status = result.status
                existing_order.timestamp = result.timestamp
                updated_orders.append(existing_order)
            else:
                # 添加新订单（来自结果）
                result.order.status = result.status
                result.order.timestamp = result.timestamp
                updated_orders.append(result.order)
            
            processed_order_ids.add(order_id)
            
            # 记录成功/失败统计
            if result.status == OrderStatus.SUCCESS:
                success_count += 1
                logger.info("Order %s executed successfully: %s", order_id, result.content)
            elif result.status == OrderStatus.FAILED:
                failure_count += 1
                logger.warning("Order %s failed: %s - %s", order_id, result.content, result.error_reason)
        
        # 添加没有结果的订单（保持原状态）
        for order in orders.orders:
            if order.order_id not in processed_order_ids:
                updated_orders.append(order)
        
        # 记录处理统计
        if success_count > 0 or failure_count > 0:
            logger.info("Order results processed: %d success, %d failure", 
                       success_count, failure_count)
        
        return OrderList(orders=updated_orders)
    
    def _process_order_expiry_and_cleanup(self, orders: OrderList, current_time: datetime) -> OrderList:
        """处理订单过期和清理
        
        Args:
            orders: 当前订单列表
            current_time: 当前时间
            
        Returns:
            OrderList: 处理后的订单列表
        """
        valid_orders = []
        expired_count = 0
        removed_count = 0
        
        for order in orders.orders:
            order_age = current_time - order.timestamp
            
            # 检查是否超过两周（完全删除）
            if order_age > timedelta(days=14):
                # 保存到历史记录
                self._order_history.append(order)
                removed_count += 1
                logger.debug("Removed order %s (age: %s)", order.order_id, order_age)
                continue
            
            # 检查是否超过一周（标记为过期）
            if (order_age > timedelta(days=7) and 
                order.status == OrderStatus.PENDING):
                order.status = OrderStatus.EXPIRED
                expired_count += 1
                logger.debug("Expired order %s (age: %s)", order.order_id, order_age)
            
            valid_orders.append(order)
        
        # 记录清理统计
        if expired_count > 0 or removed_count > 0:
            logger.info("Order cleanup: %d expired, %d removed", expired_count, removed_count)
        
        return OrderList(orders=valid_orders)
    
    def _update_account_info(self, timestamp: datetime, order_results: ResultList, 
                           current_orders: OrderList) -> AccountData:
        """更新账户信息，考虑订单执行结果的影响
        
        Args:
            timestamp: 时间戳
            order_results: 交易结果列表
            current_orders: 当前订单列表
            
        Returns:
            AccountData: 更新后的账户信息
        """
        if self._current_observation is None:
            return self._create_initial_account(timestamp)
        
        current_account = self._current_observation.account_info
        cash_change = 0.0
        market_value_change = 0.0
        
        # 计算订单结果对账户的影响
        for result in order_results.results:
            if (result.status == OrderStatus.SUCCESS and 
                result.executed_price and result.executed_quantity):
                
                trade_amount = result.executed_price * result.executed_quantity
                fee = result.fee or 0.0
                
                # 根据订单类型确定资金流向
                if result.order.order_type == OrderType.BUY:
                    # 买入：减少现金，增加持仓市值
                    cash_change -= (trade_amount + fee)
                    market_value_change += trade_amount
                elif result.order.order_type == OrderType.SELL:
                    # 卖出：增加现金，减少持仓市值
                    cash_change += (trade_amount - fee)
                    market_value_change -= trade_amount
        
        # 计算当前持仓市值（基于市场数据）
        current_market_value = self._calculate_current_market_value()
        
        # 更新账户信息
        new_available_cash = current_account.available_cash + cash_change
        new_market_value = current_market_value
        new_total_assets = new_available_cash + new_market_value
        new_net_assets = new_total_assets  # 简化处理
        
        # 计算盈亏（基于当前市值和成本）
        total_profit_loss = self._calculate_total_profit_loss()
        today_profit_loss = self._calculate_today_profit_loss()
        
        # 计算盈亏比例
        total_profit_loss_rate = (total_profit_loss / (new_total_assets - total_profit_loss)) * 100 if (new_total_assets - total_profit_loss) > 0 else 0.0
        
        # 计算持仓比例
        position_rate = new_market_value / new_total_assets if new_total_assets > 0 else 0.0
        
        updated_account = AccountData(
            timestamp=timestamp,
            total_assets=new_total_assets,
            net_assets=new_net_assets,
            available_cash=new_available_cash,
            market_value=new_market_value,
            total_profit_loss=total_profit_loss,
            total_profit_loss_rate=total_profit_loss_rate,
            today_profit_loss=today_profit_loss,
            position_rate=position_rate
        )
        
        return updated_account
    
    def _update_positions(self, timestamp: datetime, order_results: ResultList,
                         current_orders: OrderList) -> List[PositionData]:
        """更新持仓信息，考虑订单执行结果的影响
        
        Args:
            timestamp: 时间戳
            order_results: 交易结果列表
            current_orders: 当前订单列表
            
        Returns:
            List[PositionData]: 更新后的持仓列表
        """
        if self._current_observation is None:
            return []
        
        # 复制当前持仓并创建符号到持仓的映射
        position_map = {}
        for position in self._current_observation.positions:
            position_map[position.symbol] = position.copy(update={'timestamp': timestamp})
        
        # 处理订单结果对持仓的影响
        for result in order_results.results:
            if (result.status == OrderStatus.SUCCESS and 
                result.executed_price and result.executed_quantity):
                
                symbol = result.order.symbol
                executed_quantity = result.executed_quantity
                executed_price = result.executed_price
                
                if symbol in position_map:
                    # 更新现有持仓
                    position = position_map[symbol]
                    if result.order.order_type == OrderType.BUY:
                        # 买入：增加持仓数量，重新计算成本价
                        new_quantity = position.quantity + executed_quantity
                        new_cost = ((position.quantity * position.cost_price) + 
                                   (executed_quantity * executed_price)) / new_quantity
                        
                        position.quantity = new_quantity
                        position.available_quantity = new_quantity  # 简化处理
                        position.cost_price = new_cost
                    elif result.order.order_type == OrderType.SELL:
                        # 卖出：减少持仓数量
                        new_quantity = position.quantity - executed_quantity
                        if new_quantity <= 0:
                            # 如果全部卖出，移除该持仓
                            del position_map[symbol]
                            continue
                        else:
                            position.quantity = new_quantity
                            position.available_quantity = new_quantity
                else:
                    # 新建持仓（仅限买入）
                    if result.order.order_type == OrderType.BUY:
                        position = PositionData(
                            timestamp=timestamp,
                            symbol=symbol,
                            name=result.order.symbol,  # 简化处理，实际应从市场数据获取名称
                            quantity=executed_quantity,
                            available_quantity=executed_quantity,
                            cost_price=executed_price,
                            current_price=executed_price,  # 初始当前价等于成本价
                            market_value=executed_quantity * executed_price,
                            profit_loss=0.0,
                            profit_loss_rate=0.0
                        )
                        position_map[symbol] = position
        
        # 更新所有持仓的当前价格和市值
        for symbol, position in position_map.items():
            current_price = self._get_current_stock_price(symbol)
            if current_price:
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.profit_loss = (current_price - position.cost_price) * position.quantity
                if position.cost_price > 0:
                    position.profit_loss_rate = (position.profit_loss / 
                                               (position.cost_price * position.quantity)) * 100
        
        return list(position_map.values())
    
    def _create_initial_account(self, timestamp: datetime) -> AccountData:
        """创建初始账户信息"""
        return AccountData(
            timestamp=timestamp,
            total_assets=self.config.initial_total_assets,
            net_assets=self.config.initial_net_assets,
            available_cash=self.config.initial_cash,
            market_value=0.0,
            total_profit_loss=0.0,
            total_profit_loss_rate=0.0,
            today_profit_loss=0.0,
            position_rate=0.0
        )
    
    def _create_initial_positions(self, timestamp: datetime) -> List[PositionData]:
        """创建初始持仓"""
        positions = []
        for pos_config in self.config.initial_positions:
            position = PositionData(
                timestamp=timestamp,
                symbol=pos_config.get('symbol', ''),
                name=pos_config.get('name', ''),
                quantity=pos_config.get('quantity', 0),
                available_quantity=pos_config.get('available_quantity', 0),
                cost_price=pos_config.get('cost_price', 0.0),
                current_price=pos_config.get('current_price', 0.0),
                market_value=pos_config.get('market_value', 0.0),
                profit_loss=pos_config.get('profit_loss', 0.0),
                profit_loss_rate=pos_config.get('profit_loss_rate', 0.0)
            )
            positions.append(position)
        
        return positions
    
    def _initialize_market_data(self, market_data: Dict[str, Any]) -> tuple:
        """初始化市场数据"""
        market_indices = {}
        stock_data = {}
        
        # 这里根据实际的市场数据结构进行初始化
        # 简化实现，实际使用时需要根据具体数据源调整
        for index_symbol in self.config.market_index_list:
            if index_symbol in market_data.get('indices', {}):
                market_indices[index_symbol] = market_data['indices'][index_symbol]
        
        for stock_symbol in self.config.watch_list:
            if stock_symbol in market_data.get('stocks', {}):
                stock_data[stock_symbol] = market_data['stocks'][stock_symbol]
        
        return market_indices, stock_data
    
    def _update_market_data(self, market_data: Dict[str, Any]) -> tuple:
        """更新市场数据"""
        if self._current_observation is None:
            return {}, {}
        
        # 合并现有市场数据和新数据
        updated_indices = self._current_observation.market_indices.copy()
        updated_stocks = self._current_observation.stock_data.copy()
        
        # 更新指数数据
        for symbol, index_data in market_data.get('indices', {}).items():
            if symbol in self.config.market_index_list:
                updated_indices[symbol] = index_data
        
        # 更新股票数据
        for symbol, stock_data in market_data.get('stocks', {}).items():
            if symbol in self.config.watch_list:
                updated_stocks[symbol] = stock_data
        
        return updated_indices, updated_stocks
    
    def _get_market_status(self, timestamp: datetime) -> str:
        """获取市场状态"""
        # 简化实现 - 实际应该根据时间判断市场开市/收市
        hour = timestamp.hour
        if 9 <= hour < 15:
            return "open"
        else:
            return "closed"
    
    def _calculate_current_market_value(self) -> float:
        """计算当前持仓总市值"""
        if self._current_observation is None:
            return 0.0
        
        total_value = 0.0
        for position in self._current_observation.positions:
            current_price = self._get_current_stock_price(position.symbol)
            if current_price:
                total_value += position.quantity * current_price
        
        return total_value
    
    def _calculate_total_profit_loss(self) -> float:
        """计算总浮动盈亏"""
        if self._current_observation is None:
            return 0.0
        
        total_pl = 0.0
        for position in self._current_observation.positions:
            current_price = self._get_current_stock_price(position.symbol)
            if current_price:
                total_pl += (current_price - position.cost_price) * position.quantity
        
        return total_pl
    
    def _calculate_today_profit_loss(self) -> float:
        """计算当日浮动盈亏"""
        # 简化实现 - 实际应该基于前一日收盘价计算
        if self._current_observation is None:
            return 0.0
        
        # 这里可以使用更复杂的逻辑计算当日盈亏
        # 当前简化返回总盈亏的一部分
        return self._calculate_total_profit_loss() * 0.1
    
    def _get_current_stock_price(self, symbol: str) -> Optional[float]:
        """获取股票当前价格"""
        if (self._current_observation and 
            symbol in self._current_observation.stock_data and
            self._current_observation.stock_data[symbol].real_time):
            return self._current_observation.stock_data[symbol].real_time.current_price
        return None
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计信息"""
        if self._current_observation is None:
            return {}
        
        orders = self._current_observation.orders.orders
        stats = {
            "total_orders": len(orders),
            "pending_orders": len([o for o in orders if o.status == OrderStatus.PENDING]),
            "success_orders": len([o for o in orders if o.status == OrderStatus.SUCCESS]),
            "expired_orders": len([o for o in orders if o.status == OrderStatus.EXPIRED]),
            "failed_orders": len([o for o in orders if o.status == OrderStatus.FAILED]),
            "cancelled_orders": len([o for o in orders if o.status == OrderStatus.CANCELLED]),
        }
        
        return stats
    
    def get_order_history(self) -> List[Order]:
        """获取已删除的订单历史"""
        return self._order_history.copy()