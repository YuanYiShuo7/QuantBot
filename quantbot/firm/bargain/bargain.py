from textwrap import dedent
from typing import Dict, Any, Optional
from datetime import datetime
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from libs.agno.agent import Agent
from libs.agno.db.sqlite import SqliteDb
from libs.agno.models.openai import OpenAIChat
from libs.agno.run import RunContext

from general.schemas.account_schema import AccountSchema, AccountData, PositionData
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
        account_str = json.dumps(account_dict, default=str)
        
        return {
            "account": account_str,
            "last_update": datetime.now().isoformat()
        }
    
    def _session_state_to_account(self, session_state: Dict[str, Any]) -> Optional[AccountSchema]:
        """从session_state恢复账户数据"""
        try:
            account_str = session_state.get("account")
            if account_str:
                account_dict = json.loads(account_str)
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
    
    def _process_orders(self):
        """
        交易处理方法 - 秒级执行
        检查订单是否可执行，处理订单成交逻辑
        """
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 检查待处理订单...")
            
            with self.account_lock:
                # 获取待处理订单
                pending_orders = [order for order in self.account.orders 
                                if order.status == "PENDING"]
                
                if not pending_orders:
                    print("暂无待处理订单")
                    return
                
                for order in pending_orders:
                    # 模拟订单执行检查逻辑
                    print(f"检查订单 {order.order_id}: {order.symbol} {order.order_type}")
                    
                    # 模拟订单执行（50%概率执行）
                    import random
                    if random.random() > 0.5:
                        self._execute_order(order)
                        print(f"订单 {order.order_id} 已执行")
                    else:
                        print(f"订单 {order.order_id} 等待执行条件")
                    
        except Exception as e:
            print(f"处理订单时发生错误: {e}")
    
    def _execute_order(self, order):
        """
        执行订单
        Demo方法 - 实际需要接入交易接口
        """
        try:
            # 更新订单状态为成功
            order.status = "SUCCESS"
            order.timestamp = datetime.now()
            
            # 更新账户资金和持仓
            if order.order_type == "BUY":
                self._update_account_after_buy(order)
            else:  # SELL
                self._update_account_after_sell(order)
                
        except Exception as e:
            print(f"执行订单时发生错误: {e}")
            order.status = "FAILED"
    
    def _update_account_after_buy(self, order):
        """买入订单执行后更新账户"""
        cost = order.price * order.quantity
        self.account.account_info.available_cash -= cost
        
        # 更新持仓
        position = self._find_or_create_position(order.symbol)
        position.quantity += order.quantity
        position.available_quantity += order.quantity
        position.market_value = position.quantity * order.price
        
        self._update_account_metrics()
    
    def _update_account_after_sell(self, order):
        """卖出订单执行后更新账户"""
        revenue = order.price * order.quantity
        self.account.account_info.available_cash += revenue
        
        # 更新持仓
        position = self._find_position(order.symbol)
        if position:
            position.quantity -= order.quantity
            position.available_quantity -= order.quantity
            position.market_value = position.quantity * order.price
            
            if position.quantity <= 0:
                self.account.positions.remove(position)
        
        self._update_account_metrics()
    
    def _find_or_create_position(self, symbol):
        """查找或创建持仓"""
        position = self._find_position(symbol)
        if not position:
            position = PositionData(
                timestamp=datetime.now(),
                symbol=symbol,
                name=f"股票{symbol}",
                quantity=0,
                available_quantity=0,
                cost_price=0,
                current_price=0,
                market_value=0,
                profit_loss=0,
                profit_loss_rate=0
            )
            self.account.positions.append(position)
        return position
    
    def _find_position(self, symbol):
        """查找持仓"""
        for position in self.account.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _update_account_metrics(self):
        """更新账户指标"""
        # 计算总市值
        total_market_value = sum(position.market_value for position in self.account.positions)
        self.account.account_info.market_value = total_market_value
        
        # 计算总资产
        self.account.account_info.total_assets = (
            self.account.account_info.available_cash + total_market_value
        )
        
        # 计算持仓比例
        if self.account.account_info.total_assets > 0:
            self.account.account_info.position_rate = (
                total_market_value / self.account.account_info.total_assets * 100
            )
    
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
        "agent_action_interval": 30     # 30秒执行一次Agent分析（demo用短时间）
    }
    
    bargain_system = Bargain(config)
    bargain_system.run()


if __name__ == "__main__":
    main()