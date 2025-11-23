from textwrap import dedent
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import akshare as ak

from libs.agno.agent import Agent
from libs.agno.db.sqlite import SqliteDb
from libs.agno.models.openai import OpenAIChat
from libs.agno.run import RunContext

from general.schemas.account_schema import AccountSchema, AccountData, PositionData, OrderSchema, OrderStatus
from tools.order_tools import OrderTools 
from tools.watchlist_tools import WatchListTools

class Bargain:
    """交易系统核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化交易系统
        
        Args:
            config: 配置字典，包含模型配置、数据库路径等
        """
        self.config = config
        self.account = self._initialize_account()
        self.agent = self._create_agent()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.account_lock = threading.Lock()  # 账户操作锁
        
        # 配置参数
        self.order_check_interval = config.get("order_check_interval", 5)  # 秒级
        self.agent_action_interval = config.get("agent_action_interval", 60)  # 分钟级
        
        # 交易费用配置
        self.commission_rate = config.get("commission_rate", 0.0003)  # 佣金费率
        self.stamp_duty_rate = config.get("stamp_duty_rate", 0.001)   # 印花税率（卖出收取）
        self.transfer_fee_rate = config.get("transfer_fee_rate", 0.00002)  # 过户费率
        self.order_expiry_days = config.get("order_expiry_days", 7)   # 订单过期天数
        self.order_cleanup_days = config.get("order_cleanup_days", 14)  # 订单清理天数
    
    def _initialize_account(self) -> AccountSchema:
        """初始化账户数据"""
        return AccountSchema(
            timestamp=datetime.now(),
            account_info=AccountData(
                timestamp=datetime.now(),
                total_assets=100000.0,  # 初始总资产10万
                net_assets=100000.0,    # 初始净资产10万
                available_cash=100000.0, # 初始可用资金10万
                market_value=0.0,       # 初始持仓市值为0
                total_profit_loss=0.0,  # 初始总盈亏为0
                total_profit_loss_rate=0.0,  # 初始盈亏比例为0
                position_rate=0.0       # 初始持仓比例为0
            ),
            positions=[],      # 初始持仓为空
            orders=[],         # 初始订单为空
            watch_list=[       # 默认自选股
                "000001",  # 平安银行
                "000002",  # 万科A
                "600519",  # 贵州茅台
            ],
            market_index_list=[  # 默认关注指数
                "000001",  # 上证指数
                "399001",  # 深证成指
                "399006",  # 创业板指
            ]
        )
    
    def _account_to_session_state(self) -> Dict[str, Any]:
        """将账户数据转换为session_state格式"""
        account_dict = self.account.dict()
        
        return {
            "account_dict": account_dict,
            "last_update": datetime.now().isoformat()
        }

    def _session_state_to_account(self, session_state: Dict[str, Any]) -> Optional[AccountSchema]:
        """从session_state恢复账户数据"""
        try:
            account_dict = session_state.get("account_dict")
            if account_dict:
                return AccountSchema(**account_dict)
            return None
        except Exception as e:
            print(f"恢复账户数据时发生错误: {e}")
            return None
    
    def _create_agent(self) -> Agent:
        """创建交易助手Agent"""
        return Agent(
            model=OpenAIChat(
                id=self.config.get("model_id", "gpt-5o"),
                api_key=self.config.get("api_key")
            ),
            db=SqliteDb(
                db_file=self.config.get("db_file", "tmp/bargain.db")
            ),
            session_state=self._account_to_session_state(),
            tools=[
                OrderTools(),
                WatchListTools()
            ],
            instructions=dedent("""\
                你是一个专业的A股交易员。
                基于当前市场情况和账户状态，分析交易机会并执行交易决策。
                每次分析都要基于最新的账户数据。
            """),
            add_session_state_to_context=True,
            markdown=True,
            show_tool_calls=True,
            debug_mode=self.config.get("debug", False)
        )
    
    def _get_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        使用akshare获取股票实时盘口数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            实时数据字典，包含买卖盘口信息
        """
        try:
            # 获取实时数据
            stock_data = ak.stock_zh_a_spot_em()
            stock_info = stock_data[stock_data['代码'] == symbol]
            
            if stock_info.empty:
                print(f"未找到股票 {symbol} 的实时数据")
                return None
            
            # 提取盘口数据
            realtime_data = {
                'symbol': symbol,
                'name': stock_info['名称'].iloc[0],
                'current_price': float(stock_info['最新价'].iloc[0]),
                'high': float(stock_info['最高'].iloc[0]),
                'low': float(stock_info['最低'].iloc[0]),
                'bid1_price': float(stock_info['买一价'].iloc[0]),
                'bid1_volume': int(stock_info['买一量'].iloc[0]),
                'ask1_price': float(stock_info['卖一价'].iloc[0]),
                'ask1_volume': int(stock_info['卖一量'].iloc[0]),
                'volume': int(stock_info['成交量'].iloc[0]),
                'amount': float(stock_info['成交额'].iloc[0]),
                'timestamp': datetime.now()
            }
            
            return realtime_data
            
        except Exception as e:
            print(f"获取股票 {symbol} 实时数据失败: {e}")
            return None
    
    def _cleanup_expired_orders(self, timestamp: datetime):
        """清理过期订单（超过两周）"""
        cleanup_threshold = timestamp - timedelta(days=self.order_cleanup_days)
        self.account.orders = [
            order for order in self.account.orders 
            if order.timestamp > cleanup_threshold or order.status == OrderStatus.PENDING
        ]
    
    def _is_order_expired(self, order: OrderSchema, timestamp: datetime) -> bool:
        """检查订单是否过期（超过一周）"""
        expiry_threshold = timestamp - timedelta(days=self.order_expiry_days)
        return order.timestamp < expiry_threshold
    
    def _process_orders(self):
        """
        交易处理方法 - 秒级执行
        检查订单是否可执行，处理订单成交逻辑
        """
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 检查待处理订单...")
            
            with self.account_lock:
                timestamp = datetime.now()
                
                # 清理过期订单
                self._cleanup_expired_orders(timestamp)
                
                # 获取待处理订单
                pending_orders = [order for order in self.account.orders 
                                if order.status == OrderStatus.PENDING]
                
                if not pending_orders:
                    print("暂无待处理订单")
                    return
                
                print(f"发现 {len(pending_orders)} 个待处理订单")
                
                for order in pending_orders:
                    # 检查订单是否过期
                    if self._is_order_expired(order, timestamp):
                        order.status = OrderStatus.EXPIRED
                        print(f"订单已过期: {order.order_id}")
                        continue
                    
                    # 尝试执行订单
                    self._try_execute_order(order, timestamp)
                    
                # 更新账户资产数据
                self._update_account_assets(timestamp)
                    
        except Exception as e:
            print(f"处理订单时发生错误: {e}")
    
    def _try_execute_order(self, order: OrderSchema, timestamp: datetime):
        """尝试执行单个订单"""
        try:
            # 获取股票实时数据
            realtime_data = self._get_realtime_data(order.symbol)
            if not realtime_data:
                print(f"股票数据不存在: {order.symbol}")
                return
            
            # 检查价格是否在当日波动范围内
            if not (realtime_data['low'] <= order.price <= realtime_data['high']):
                print(f"订单价格不在波动范围内: {order.price} not in [{realtime_data['low']}, {realtime_data['high']}]")
                return
            
            # 检查订单执行条件
            if order.order_type == "BUY":
                # 买入订单：订单价格 >= 卖一价 才能成交
                if order.price >= realtime_data['ask1_price']:
                    self._execute_buy_order(order, realtime_data, timestamp)
                else:
                    print(f"买入订单价格 {order.price} < 卖一价 {realtime_data['ask1_price']}，等待")
            else:  # SELL
                # 卖出订单：订单价格 <= 买一价 才能成交
                if order.price <= realtime_data['bid1_price']:
                    self._execute_sell_order(order, realtime_data, timestamp)
                else:
                    print(f"卖出订单价格 {order.price} > 买一价 {realtime_data['bid1_price']}，等待")
                    
        except Exception as e:
            print(f"执行订单失败 {order.order_id}: {str(e)}")
    
    def _execute_buy_order(self, order: OrderSchema, realtime_data: Dict[str, Any], timestamp: datetime):
        """执行买入订单"""
        try:
            # 计算交易金额和费用
            total_cost = order.price * order.quantity
            commission = total_cost * self.commission_rate
            transfer_fee = total_cost * self.transfer_fee_rate
            total_amount = total_cost + commission + transfer_fee
            
            # 检查资金是否足够
            if self.account.account_info.available_cash < total_amount:
                print(f"资金不足，无法执行买入订单: 需要{total_amount:.2f}，可用{self.account.account_info.available_cash:.2f}")
                return
            
            # 扣除资金
            self.account.account_info.available_cash -= total_amount
            
            # 更新或创建持仓
            position = self._get_position(order.symbol)
            if position:
                # 更新现有持仓（成本价按加权平均计算）
                old_value = position.cost_price * position.quantity
                new_value = order.price * order.quantity
                total_quantity = position.quantity + order.quantity
                position.cost_price = (old_value + new_value) / total_quantity
                position.quantity = total_quantity
                position.available_quantity = total_quantity
                position.current_price = realtime_data['current_price']
                position.market_value = position.quantity * realtime_data['current_price']
            else:
                # 创建新持仓
                position = PositionData(
                    timestamp=timestamp,
                    symbol=order.symbol,
                    name=realtime_data['name'],
                    quantity=order.quantity,
                    available_quantity=order.quantity,
                    cost_price=order.price,
                    current_price=realtime_data['current_price'],
                    market_value=order.quantity * realtime_data['current_price'],
                    profit_loss=0.0,
                    profit_loss_rate=0.0
                )
                self.account.positions.append(position)
            
            # 更新订单状态
            order.status = OrderStatus.SUCCESS
            order.timestamp = timestamp
            
            print(f"✅ 买入订单执行成功: {order.symbol} {order.quantity}股 @ {order.price}")
            
        except Exception as e:
            print(f"执行买入订单失败: {str(e)}")
    
    def _execute_sell_order(self, order: OrderSchema, realtime_data: Dict[str, Any], timestamp: datetime):
        """执行卖出订单"""
        try:
            # 检查持仓是否足够
            position = self._get_position(order.symbol)
            if not position or position.available_quantity < order.quantity:
                available_qty = position.available_quantity if position else 0
                print(f"持仓不足，无法执行卖出订单: 需要{order.quantity}，可用{available_qty}")
                return
            
            # 计算交易金额和费用
            total_amount = order.price * order.quantity
            commission = total_amount * self.commission_rate
            stamp_duty = total_amount * self.stamp_duty_rate  # 卖出收取印花税
            transfer_fee = total_amount * self.transfer_fee_rate
            net_amount = total_amount - commission - stamp_duty - transfer_fee
            
            # 更新持仓
            position.quantity -= order.quantity
            position.available_quantity -= order.quantity
            position.current_price = realtime_data['current_price']
            position.market_value = position.quantity * realtime_data['current_price']
            
            # 如果持仓为0，移除该持仓
            if position.quantity <= 0:
                self.account.positions.remove(position)
            
            # 增加资金
            self.account.account_info.available_cash += net_amount
            
            # 更新订单状态
            order.status = OrderStatus.SUCCESS
            order.timestamp = timestamp
            
            print(f"✅ 卖出订单执行成功: {order.symbol} {order.quantity}股 @ {order.price}")
            
        except Exception as e:
            print(f"执行卖出订单失败: {str(e)}")
    
    def _get_position(self, symbol: str) -> Optional[PositionData]:
        """获取指定股票的持仓"""
        for position in self.account.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _update_account_assets(self, timestamp: datetime):
        """更新账户资产数据"""
        try:
            # 更新持仓的当前价格和市值
            total_market_value = 0.0
            total_profit_loss = 0.0
            total_cost = 0.0
            
            for position in self.account.positions:
                # 更新持仓的当前价格（从实时数据获取）
                realtime_data = self._get_realtime_data(position.symbol)
                if realtime_data:
                    current_price = realtime_data['current_price']
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
            account_info = self.account.account_info
            account_info.timestamp = timestamp
            account_info.market_value = total_market_value
            account_info.total_assets = account_info.available_cash + total_market_value
            account_info.net_assets = account_info.total_assets
            account_info.total_profit_loss = total_profit_loss
            account_info.total_profit_loss_rate = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0.0
            account_info.position_rate = (total_market_value / account_info.total_assets * 100) if account_info.total_assets > 0 else 0.0
            
        except Exception as e:
            print(f"更新账户资产数据失败: {str(e)}")
    
    def _agent_action(self):
        """
        Agent Action方法 - 分钟级执行
        基于市场情况和账户状态进行交易决策
        """
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent执行交易分析...")
            
            # 在Agent运行前，使用最新的账户状态创建新的session state
            session_state = self._account_to_session_state()
            
            # 运行Agent，传入最新的账户状态
            response = self.agent.run(
                session_state=session_state,
                input="基于当前市场情况和账户状态，分析交易机会并给出交易建议"
            )
            
            # Agent运行后，从session state获取更新后的账户数据
            updated_session_state = self.agent.get_session_state()
            updated_account = self._session_state_to_account(updated_session_state)
            
            # 使用锁来安全地更新账户
            if updated_account:
                with self.account_lock:
                    self.account = updated_account
                    print("✅ 账户状态已从Agent更新")
            
        except Exception as e:
            print(f"Agent Action执行错误: {e}")
    
    def start_trading_loop(self):
        """启动交易循环"""
        self.running = True
        print("启动交易系统...")
        
        # 启动订单处理循环（秒级）
        self.executor.submit(self._order_processing_loop)
        
        # 启动Agent Action循环（分钟级）
        self.executor.submit(self._agent_action_loop)
    
    def _order_processing_loop(self):
        """订单处理循环 - 秒级"""
        while self.running:
            try:
                self._process_orders()
                time.sleep(self.order_check_interval)
            except Exception as e:
                print(f"订单处理循环错误: {e}")
                time.sleep(1)
    
    def _agent_action_loop(self):
        """Agent Action循环 - 分钟级"""
        while self.running:
            try:
                self._agent_action()
                time.sleep(self.agent_action_interval)
            except Exception as e:
                print(f"Agent Action循环错误: {e}")
                time.sleep(10)
    
    def stop_trading(self):
        """停止交易"""
        self.running = False
        self.executor.shutdown(wait=False)
        print("交易系统已停止")
    
    def run(self) -> None:
        """运行交易系统"""
        try:
            self.start_trading_loop()
            
            # 保持主线程运行
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n收到停止信号...")
            self.stop_trading()
        except Exception as e:
            print(f"交易系统运行错误: {e}")
            self.stop_trading()


def main():
    """主函数"""
    config = {
        "model_id": "gpt-4o",
        "api_key": "your_openai_api_key_here",
        "db_file": "tmp/bargain_demo.db",
        "debug": True,
        "order_check_interval": 5,      # 5秒检查一次订单
        "agent_action_interval": 120,    # 120秒执行一次Agent分析（demo用短时间）
        "commission_rate": 0.0003,      # 佣金费率
        "stamp_duty_rate": 0.001,       # 印花税率
        "transfer_fee_rate": 0.00002,   # 过户费率
        "order_expiry_days": 7,         # 订单过期天数
        "order_cleanup_days": 14        # 订单清理天数
    }
    
    bargain_system = Bargain(config)
    bargain_system.run()


if __name__ == "__main__":
    main()