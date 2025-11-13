from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

class Simulator(SimulatorInterface):
    """模拟器实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 从配置获取参数
        self.config_file_path = self.config.get('config_file_path', 'simulator_config.json')
        self.training_interval = self.config.get('training_interval', 10)  # 每多少轮训练一次
        self.episodes_per_config = self.config.get('episodes_per_config', 1)  # 每个配置跑多少轮次
        
        # 状态变量
        self.current_episode = 0
        self.total_episodes = 0
        self.configs = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("模拟器初始化完成")
        self.logger.info(f"配置文件: {self.config_file_path}")
        self.logger.info(f"训练间隔: {self.training_interval} 轮")
        self.logger.info(f"每配置轮次: {self.episodes_per_config}")
    
    def run(self) -> None:
        """运行模拟器"""
        try:
            # 1. 加载配置文件
            self._load_configs()
            
            # 2. 遍历所有配置
            for config_index, config_data in enumerate(self.configs):
                self.logger.info(f"开始执行配置 {config_index + 1}/{len(self.configs)}")
                
                # 3. 每个配置运行指定轮次
                for episode in range(self.episodes_per_config):
                    self.current_episode += 1
                    self.total_episodes += 1
                    
                    self.logger.info(f"开始第 {self.current_episode} 轮模拟 (配置{config_index + 1}-轮次{episode + 1})")
                    
                    # 4. 初始化各个组件
                    components = self._initialize_components(config_data)
                    
                    # 5. 运行单轮模拟
                    self._run_single_episode(components)
                    
                    # 6. 计算奖励
                    components['reward'].calculate_score()
                    
                    # 7. 持久化数据
                    components['reward'].persist_data()
                    
                    # 8. 检查是否需要进行训练
                    if (config_data.get('should_train', False) and 
                        self.current_episode % self.training_interval == 0):
                        self._train_model(components['trainer'], components['llm_agent'])
                    
                    self.logger.info(f"第 {self.current_episode} 轮模拟完成")
            
            self.logger.info("所有模拟轮次完成")
            
        except Exception as e:
            self.logger.error(f"模拟器运行失败: {str(e)}")
            raise
    
    def _load_configs(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                self.configs = json.load(f)
            
            self.logger.info(f"成功加载 {len(self.configs)} 个配置")
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise
    
    def _initialize_components(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """初始化各个组件"""
        components = {}
        
        try:
            # 初始化环境
            from environment.base import Environment
            components['environment'] = Environment(config_data.get('environment', {}))
            
            # 初始化计时器
            from timer.base import Timer
            components['timer'] = Timer(config_data.get('timer', {}))
            
            # 初始化账户
            from account.base import Account
            components['account'] = Account(config_data.get('account', {}))
            
            # 初始化交易所
            from exchange.base import Exchange
            components['exchange'] = Exchange(config_data.get('exchange', {}))
            
            # 初始化训练器（如果配置了训练）
            if config_data.get('should_train', False):
                from trainer.base import Trainer
                components['trainer'] = Trainer(config_data.get('trainer', {}))
            
            # 初始化LLM代理
            from llm_agent.base import LLMAgent
            components['llm_agent'] = LLMAgent(config_data.get('llm_agent', {}))
            
            # 初始化市场
            from market.base import Market
            components['market'] = Market(config_data.get('market', {}))
            
            # 初始化奖励系统
            from reward.base import Reward
            components['reward'] = Reward(config_data.get('reward', {}))
            
            self.logger.info("所有组件初始化完成")
            return components
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def _run_single_episode(self, components: Dict[str, Any]) -> None:
        """运行单轮模拟"""
        environment = components['environment']
        timer = components['timer']
        account = components['account']
        exchange = components['exchange']
        llm_agent = components['llm_agent']
        market = components['market']
        reward = components['reward']
        
        # 运行直到终止条件
        is_finished = False
        step_count = 0
        
        while not is_finished:
            step_count += 1
            self.logger.debug(f"第 {step_count} 步")
            
            is_finished = environment.step(timer, account, exchange, llm_agent, market, reward)
            
            # 安全限制，防止无限循环
            if step_count > 1000:
                self.logger.warning("达到最大步数限制，强制终止")
                break
        
        self.logger.info(f"单轮模拟完成，共执行 {step_count} 步")
    
    def _train_model(self, trainer, llm_agent) -> None:
        """训练模型"""
        try:
            self.logger.info(f"开始第 {self.current_episode} 轮训练")
            
            # 调用训练器的训练方法
            trainer.train(llm_agent)
            
            self.logger.info("训练完成")
            
        except Exception as e:
            self.logger.error(f"训练失败: {str(e)}")