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
        
        # 从配置获取初始参数
        self.initial_cash = self.config.get('initial_cash', 1000000.0)  # 默认100万
        self.watch_list = self.config.get('watch_list', [])
        self.market_index_list = self.config.get('market_index_list', [])
        
        # 公共账户对象
        self.account = self._initialize_account()
        
        # 缓存配置
        self.cache_dir = Path(self.config.get('cache_dir', './account_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"账户接口初始化完成，初始资金: {self.initial_cash}")
        self.logger.info(f"关注股票: {self.watch_list}")
        self.logger.info(f"关注指数: {self.market_index_list}")
    
    def _initialize_account(self) -> AccountSchema:
        """初始化账户数据"""
        # 创建账户基础信息
        account_data = AccountData(
            timestamp=datetime.now(),
            total_assets=self.initial_cash,
            net_assets=self.initial_cash,
            available_cash=self.initial_cash,
            market_value=0.0,
            total_profit_loss=0.0,
            total_profit_loss_rate=0.0,
            today_profit_loss=0.0,
            position_rate=0.0,
            margin_ratio=None
        )
        
        # 创建完整的账户schema
        account_schema = AccountSchema(
            timestamp=datetime.now(),
            account_info=account_data,
            positions=[],  # 初始无持仓
            orders=[],     # 初始无订单
            watch_list=self.watch_list,
            market_index_list=self.market_index_list,
            trading_enabled=True  # 默认允许交易
        )
        
        return account_schema
    
    def format_account_info_for_prompt(self) -> str:
        """
        将账户信息 AccountSchema 转换为 Agent 的 Prompt 文本
        """
        try:
            account = self.account
            
            # 构建账户基本信息
            prompt_lines = [
                "=== 账户信息 ===",
                f"更新时间: {account.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"总资产: ¥{account.account_info.total_assets:,.2f}",
                f"净资产: ¥{account.account_info.net_assets:,.2f}",
                f"可用资金: ¥{account.account_info.available_cash:,.2f}",
                f"持仓市值: ¥{account.account_info.market_value:,.2f}",
                f"持仓比例: {account.account_info.position_rate:.1f}%",
                f"总盈亏: ¥{account.account_info.total_profit_loss:,.2f} ({account.account_info.total_profit_loss_rate:+.2f}%)",
                f"当日盈亏: ¥{account.account_info.today_profit_loss:,.2f}",
                ""
            ]
            
            # 构建持仓信息
            if account.positions:
                prompt_lines.extend([
                    "=== 持仓信息 ===",
                    f"{'代码':<8} {'名称':<10} {'数量':>8} {'成本':>10} {'现价':>10} {'市值':>12} {'盈亏':>12} {'盈亏率':>8}",
                    "-" * 90
                ])
                
                for position in account.positions:
                    prompt_lines.append(
                        f"{position.symbol:<8} {position.name:<10} "
                        f"{position.quantity:>8,d} {position.cost_price:>10.2f} "
                        f"{position.current_price:>10.2f} {position.market_value:>12,.2f} "
                        f"{position.profit_loss:>12,.2f} {position.profit_loss_rate:>+7.2f}%"
                    )
            else:
                prompt_lines.append("=== 持仓信息 ===")
                prompt_lines.append("暂无持仓")
            
            prompt_lines.append("")
            
            # 构建订单信息
            if account.orders:
                prompt_lines.extend([
                    "=== 当前订单 ===",
                    f"{'订单ID':<12} {'代码':<8} {'类型':<6} {'价格':>8} {'数量':>8} {'状态':<12} {'时间':<10}",
                    "-" * 70
                ])
                
                for order in account.orders:
                    prompt_lines.append(
                        f"{order.order_id:<12} {order.symbol:<8} {order.order_type:<6} "
                        f"{order.price:>8.2f} {order.quantity:>8,d} {order.status:<12} "
                        f"{order.timestamp.strftime('%H:%M:%S'):<10}"
                    )
            else:
                prompt_lines.append("=== 当前订单 ===")
                prompt_lines.append("暂无订单")
            
            prompt_lines.append("")
            
            # 构建关注列表
            if account.watch_list:
                prompt_lines.append(f"=== 自选股列表 ===")
                prompt_lines.append(", ".join(account.watch_list))
            
            if account.market_index_list:
                prompt_lines.append(f"=== 关注指数 ===")
                prompt_lines.append(", ".join(account.market_index_list))
            
            prompt_lines.append("")
            prompt_lines.append(f"交易状态: {'允许交易' if account.trading_enabled else '禁止交易'}")
            
            return "\n".join(prompt_lines)
            
        except Exception as e:
            self.logger.error(f"格式化账户信息失败: {str(e)}")
            return f"账户信息格式化失败: {str(e)}"
    
    def get_account_info(self) -> AccountSchema:
        """
        获取完整的账户信息
        """
        # 更新账户时间戳
        self.account.timestamp = datetime.now()
        self.account.account_info.timestamp = datetime.now()
        
        # 更新持仓时间戳
        for position in self.account.positions:
            position.timestamp = datetime.now()
        
        return self.account
    
    def add_order(self, order_forms: List[OrderFormSchema], timestamp: datetime, order_id: str) -> OrderResultSchema:
        """
        添加新订单
        """
        # 这里实现具体的下单逻辑
        # 由于没有具体的交易接口，这里返回一个模拟的成功结果
        try:
            # 模拟订单执行
            order_result = OrderResultSchema(
                order_id=order_id,
                symbol=order_forms[0].symbol if order_forms else "",
                order_type=order_forms[0].order_type if order_forms else "BUY",
                status="success",
                timestamp=timestamp,
                executed_quantity=order_forms[0].quantity if order_forms else 0,
                executed_price=order_forms[0].price if order_forms else 0,
                executed_amount=(order_forms[0].quantity * order_forms[0].price) if order_forms else 0,
                commission=5.0,  # 模拟手续费
                stamp_duty=0.0,
                transfer_fee=0.1,
                total_fee=5.1,
                net_amount=(order_forms[0].quantity * order_forms[0].price + 5.1) if order_forms else 0,
                error_message=None,
                error_code=None
            )
            
            # 创建订单记录
            for i, order_form in enumerate(order_forms):
                individual_order_id = f"{order_id}_{i}" if len(order_forms) > 1 else order_id
                order = OrderSchema(
                    order_id=individual_order_id,
                    symbol=order_form.symbol,
                    order_type=order_form.order_type,
                    price=order_form.price,
                    quantity=order_form.quantity,
                    timestamp=timestamp,
                    status="success"
                )
                self.account.orders.append(order)
            
            self.logger.info(f"订单添加成功: {order_id}")
            return order_result
            
        except Exception as e:
            self.logger.error(f"订单添加失败: {str(e)}")
            return OrderResultSchema(
                order_id=order_id,
                symbol=order_forms[0].symbol if order_forms else "",
                order_type=order_forms[0].order_type if order_forms else "BUY",
                status="failed",
                timestamp=timestamp,
                error_message=str(e),
                error_code="ORDER_ADD_ERROR"
            )
    
    def cancel_order(self, order_id: str) -> OrderResultSchema:
        """
        取消指定订单
        """
        try:
            # 查找订单
            order_to_cancel = None
            for order in self.account.orders:
                if order.order_id == order_id:
                    order_to_cancel = order
                    break
            
            if not order_to_cancel:
                return OrderResultSchema(
                    order_id=order_id,
                    symbol="",
                    order_type="BUY",
                    status="failed",
                    timestamp=datetime.now(),
                    error_message=f"订单 {order_id} 不存在",
                    error_code="ORDER_NOT_FOUND"
                )
            
            # 更新订单状态
            order_to_cancel.status = "cancelled"
            
            self.logger.info(f"订单取消成功: {order_id}")
            return OrderResultSchema(
                order_id=order_id,
                symbol=order_to_cancel.symbol,
                order_type=order_to_cancel.order_type,
                status="cancelled",
                timestamp=datetime.now(),
                error_message=None,
                error_code=None
            )
            
        except Exception as e:
            self.logger.error(f"订单取消失败: {str(e)}")
            return OrderResultSchema(
                order_id=order_id,
                symbol="",
                order_type="BUY",
                status="failed",
                timestamp=datetime.now(),
                error_message=str(e),
                error_code="ORDER_CANCEL_ERROR"
            )
    
    def update_account_info(self, order_results: List[OrderResultSchema]) -> AccountSchema:
        """
        更新账户信息
        """
        try:
            # 根据订单结果更新账户状态
            for result in order_results:
                if result.status == "success":
                    # 更新账户资金和持仓
                    self._apply_successful_order(result)
            
            # 重新计算账户统计信息
            self._recalculate_account_stats()
            
            self.logger.info("账户信息更新成功")
            return self.account
            
        except Exception as e:
            self.logger.error(f"账户信息更新失败: {str(e)}")
            return self.account
    
    def _apply_successful_order(self, order_result: OrderResultSchema) -> None:
        """应用成功的订单到账户"""
        if order_result.order_type == "BUY":
            # 买入操作：减少现金，增加持仓
            self.account.account_info.available_cash -= order_result.net_amount
            
            # 查找或创建持仓
            position = self._find_or_create_position(order_result.symbol, "未知股票")
            position.quantity += int(order_result.executed_quantity)
            position.available_quantity += int(order_result.executed_quantity)
            # 更新成本价（加权平均）
            total_cost = (position.cost_price * (position.quantity - order_result.executed_quantity) + 
                         order_result.executed_amount)
            position.cost_price = total_cost / position.quantity
            
        else:  # SELL
            # 卖出操作：增加现金，减少持仓
            self.account.account_info.available_cash += (order_result.executed_amount - order_result.total_fee)
            
            # 减少持仓
            position = self._find_position(order_result.symbol)
            if position:
                position.quantity -= int(order_result.executed_quantity)
                position.available_quantity -= int(order_result.executed_quantity)
                
                # 如果持仓为0，移除该持仓
                if position.quantity <= 0:
                    self.account.positions = [p for p in self.account.positions if p.symbol != order_result.symbol]
    
    def _find_or_create_position(self, symbol: str, name: str) -> PositionData:
        """查找或创建持仓"""
        position = self._find_position(symbol)
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
            self.account.positions.append(position)
        return position
    
    def _find_position(self, symbol: str) -> Optional[PositionData]:
        """查找持仓"""
        for position in self.account.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _recalculate_account_stats(self) -> None:
        """重新计算账户统计信息"""
        # 计算持仓市值和盈亏
        total_market_value = 0.0
        total_profit_loss = 0.0
        
        for position in self.account.positions:
            # 这里需要从市场数据获取当前价格，暂时使用成本价
            position.current_price = position.cost_price  # 应该从市场接口获取实时价格
            position.market_value = position.quantity * position.current_price
            position.profit_loss = (position.current_price - position.cost_price) * position.quantity
            position.profit_loss_rate = (position.current_price / position.cost_price - 1) * 100 if position.cost_price > 0 else 0
            
            total_market_value += position.market_value
            total_profit_loss += position.profit_loss
        
        # 更新账户信息
        self.account.account_info.market_value = total_market_value
        self.account.account_info.total_assets = self.account.account_info.available_cash + total_market_value
        self.account.account_info.total_profit_loss = total_profit_loss
        self.account.account_info.total_profit_loss_rate = (total_profit_loss / (self.account.account_info.total_assets - total_profit_loss)) * 100 if (self.account.account_info.total_assets - total_profit_loss) > 0 else 0
        self.account.account_info.position_rate = (total_market_value / self.account.account_info.total_assets) * 100 if self.account.account_info.total_assets > 0 else 0
    
    def save_account_state(self, filepath: str = None) -> bool:
        """保存账户状态到文件"""
        try:
            if filepath is None:
                filepath = self.cache_dir / 'account_state.pkl'
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.account, f)
            
            self.logger.info(f"账户状态已保存: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存账户状态失败: {str(e)}")
            return False
    
    def load_account_state(self, filepath: str = None) -> bool:
        """从文件加载账户状态"""
        try:
            if filepath is None:
                filepath = self.cache_dir / 'account_state.pkl'
            
            if not Path(filepath).exists():
                self.logger.warning(f"账户状态文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                self.account = pickle.load(f)
            
            self.logger.info(f"账户状态已加载: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载账户状态失败: {str(e)}")
            return False