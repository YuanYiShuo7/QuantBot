from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from general.schemas.action_schema import ActionSchemaUnion, AddOrderActionSchema, CancelOrderActionSchema, NoneActionSchema, ActionType
from general.schemas.order_schema import OrderFormSchema, OrderType
from general.interfaces.llm_agent_interface import LLMAgentInterface
from account.base import Account
from market.base import Market

class LLMAgent(LLMAgentInterface):
    """默认LLM Agent实现类 - 使用HuggingFace本地模型"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 模型配置
        self.model_path = self.config.get('model_path', 'deepseek-ai/DeepSeek-V3')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_new_tokens = self.config.get('max_new_tokens', 1024)
        
        # 系统提示词
        self.system_prompt = self.config.get('system_prompt', self._get_default_system_prompt())
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        self.logger.info(f"LLM Agent初始化完成，模型路径: {self.model_path}")
        self.logger.info(f"设备: {self.device}")
    
    def _load_model(self):
        """加载本地模型"""
        try:
            self.logger.info("正在加载模型...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 如果分词器没有pad_token，设置为eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
            # 生成配置
            self.generation_config = GenerationConfig(
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            self.logger.info("模型加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词 - A股交易专用"""
        return """你是一个专业的A股量化交易员。请基于提供的市场信息和账户状态，做出符合A股交易规则的理性投资决策。

A股交易规则：
1. 交易时间：工作日 9:30-11:30, 13:00-15:00
2. 交易单位：买入数量必须为100股及其整数倍（1手=100股）
3. 价格限制：涨跌幅限制为±10%（ST股为±5%）
4. T+1制度：当日买入的股票下一交易日才能卖出

风险控制原则：
1. 单只股票持仓不超过总资产的20%
2. 单次交易金额不超过可用资金的30%
3. 保持投资组合的适度分散

输出要求：
- 必须使用JSON格式输出
- 仅按照示例输出JSON代码块，不包含任何其他文本
- 可以同时执行多个订单操作
- 每个操作必须是独立的JSON对象
- 确保所有数值字段类型正确
- 数量必须是100的整数倍
- 价格保留2位小数

输出示例：

示例1 - 单个买入操作：
{
  "reasoning": "基于技术分析，平安银行突破关键阻力位，MACD金叉，建议买入",
  "action_type": "ADD_ORDER",
  "symbol": "000001",
  "order_type": "BUY",
  "price": 12.50,
  "quantity": 500
}

示例2 - 多个操作：
[
  {
    "reasoning": "贵州茅台技术面出现顶背离，建议部分获利了结",
    "action_type": "ADD_ORDER", 
    "symbol": "600519",
    "order_type": "SELL",
    "price": 1650.00,
    "quantity": 100
  },
  {
    "reasoning": "宁德时代回调至支撑位，估值合理，建议买入",
    "action_type": "ADD_ORDER",
    "symbol": "300750", 
    "order_type": "BUY",
    "price": 185.50,
    "quantity": 200
  }
]

示例3 - 取消订单：
{
  "reasoning": "市场环境变化，原买单价格偏离当前市价过大，建议取消",
  "action_type": "CANCEL_ORDER",
  "order_id": "ORD_20240115001"
}

示例4 - 无操作：
{
  "reasoning": "当前市场震荡较大，缺乏明确趋势，建议观望等待更好机会",
  "action_type": "NONE"
}

请基于当前市场状况和账户情况，输出你的交易决策："""
    
    def generate_prompt(self, account: Account, market: Market) -> str:
        """生成完整的prompt文本"""
        try:
            # 获取格式化的市场信息和账户信息
            market_info = market.format_market_info_for_prompt()
            account_info = account.format_account_info_for_prompt()
            
            # 构建完整prompt
            prompt = f"""{self.system_prompt}

当前市场信息：
{market_info}

当前账户信息：
{account_info}

请分析当前情况并输出你的交易决策（JSON格式）：
"""
            return prompt
            
        except Exception as e:
            self.logger.error(f"生成prompt失败: {str(e)}")
            return self.system_prompt
    
def generate_output(self, prompt: str) -> str:
    """基于prompt生成输出 - 单轮对话模式（简化版）"""
    try:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("模型未正确加载")
        
        # 直接构建Qwen2对话格式
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 对完整文本进行编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        # 移动到设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,  # Qwen2使用eos_token作为pad_token
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型回复部分（去除输入）
        response = generated_text[len(text):].strip()
        
        # 如果响应以<|im_end|>结尾，移除它
        if response.endswith("<|im_end|>"):
            response = response[:-10].strip()
        
        self.logger.debug(f"模型原始输出: {response[:200]}...")
        return response
        
    except Exception as e:
        self.logger.error(f"生成输出失败: {str(e)}")
        # 返回默认的无操作响应
        return json.dumps({
            "reasoning": f"系统错误，无法生成交易决策: {str(e)}",
            "action_type": "NONE"
        }, ensure_ascii=False)
    
    def parse_action(self, output: str) -> List[ActionSchemaUnion]:
        """解析输出文本为动作列表"""
        try:
            actions = []
            
            # 尝试从输出中提取所有JSON对象
            json_objects = self._extract_json_objects(output)
            
            if not json_objects:
                self.logger.warning("输出中未找到有效的JSON格式")
                # 返回默认的无操作
                return [NoneActionSchema(
                    reasoning="输出格式错误，无法解析",
                    action_type=ActionType.NONE
                )]
            
            for json_obj in json_objects:
                try:
                    action = self._parse_single_action(json_obj)
                    if action:
                        actions.append(action)
                except Exception as e:
                    self.logger.warning(f"解析单个动作失败: {str(e)}")
                    continue
            
            # 如果没有有效动作，返回无操作
            if not actions:
                actions.append(NoneActionSchema(
                    reasoning="未找到有效交易决策",
                    action_type=ActionType.NONE
                ))
            
            self.logger.info(f"成功解析 {len(actions)} 个动作")
            return actions
            
        except Exception as e:
            self.logger.error(f"解析动作失败: {str(e)}")
            # 返回默认的无操作
            return [NoneActionSchema(
                reasoning=f"动作解析失败: {str(e)}",
                action_type=ActionType.NONE
            )]
    
    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取所有JSON对象"""
        json_objects = []
        
        # 方法1: 尝试解析整个文本为JSON数组
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except:
            pass
        
        # 方法2: 使用正则表达式查找JSON对象
        # 改进的正则表达式，更好地匹配嵌套结构
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group()
                # 清理可能的格式问题
                json_str = re.sub(r',\s*}', '}', json_str)  # 修复尾随逗号
                json_str = re.sub(r',\s*]', ']', json_str)
                
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                self.logger.debug(f"JSON解析失败: {str(e)}，跳过该对象")
                continue
            except Exception as e:
                self.logger.debug(f"处理JSON对象失败: {str(e)}")
                continue
        
        return json_objects
    
    def _parse_single_action(self, data: Dict[str, Any]) -> Optional[ActionSchemaUnion]:
        """解析单个动作"""
        if not isinstance(data, dict):
            return None
        
        action_type_str = data.get('action_type', '').upper()
        reasoning = data.get('reasoning', '')
        
        try:
            if action_type_str == ActionType.ADD_ORDER.value:
                # 解析下单动作
                order_data = data.get('order_form', data)  # 兼容两种格式
                
                symbol = str(order_data.get('symbol', ''))
                order_type_str = order_data.get('order_type', 'BUY').upper()
                price = float(order_data.get('price', 0))
                quantity = int(order_data.get('quantity', 0))
                
                # 验证必需字段
                if not symbol:
                    raise ValueError("股票代码不能为空")
                
                # 验证订单类型
                try:
                    order_type = OrderType(order_type_str)
                except ValueError:
                    raise ValueError(f"无效的订单类型: {order_type_str}")
                
                # 验证价格和数量
                if price <= 0:
                    raise ValueError("价格必须大于0")
                if quantity <= 0:
                    raise ValueError("数量必须大于0")
                
                # 验证数量是否为100的整数倍
                if quantity % 100 != 0:
                    self.logger.warning(f"数量 {quantity} 不是100的整数倍，已自动调整")
                    quantity = (quantity // 100) * 100
                    if quantity == 0:
                        quantity = 100  # 最小交易单位
                
                order_form = OrderFormSchema(
                    symbol=symbol,
                    order_type=order_type,
                    price=round(price, 2),  # 保留2位小数
                    quantity=quantity
                )
                
                return AddOrderActionSchema(
                    reasoning=reasoning,
                    action_type=ActionType.ADD_ORDER,
                    order_form=order_form
                )
                
            elif action_type_str == ActionType.CANCEL_ORDER.value:
                # 解析取消订单动作
                order_id = data.get('order_id')
                if not order_id:
                    raise ValueError("取消订单必须提供order_id")
                
                return CancelOrderActionSchema(
                    reasoning=reasoning,
                    action_type=ActionType.CANCEL_ORDER,
                    order_id=str(order_id)
                )
                
            elif action_type_str == ActionType.NONE.value:
                # 无操作
                return NoneActionSchema(
                    reasoning=reasoning,
                    action_type=ActionType.NONE
                )
                
            else:
                self.logger.warning(f"未知的操作类型: {action_type_str}")
                return None
                
        except Exception as e:
            self.logger.warning(f"解析动作数据失败: {str(e)}, 数据: {data}")
            return None
    
    def generate_pipeline(self, account, market) -> Tuple[str, str, List[ActionSchemaUnion]]:
        """完整的动作生成流程"""
        try:
            # 生成prompt
            prompt = self.generate_prompt(account, market)
            self.logger.debug(f"生成的prompt长度: {len(prompt)}")
            
            # 生成输出
            output = self.generate_output(prompt)
            self.logger.debug(f"模型输出长度: {len(output)}")
            
            # 解析动作
            actions = self.parse_action(output)
            
            self.logger.info(f"成功生成 {len(actions)} 个动作")
            return prompt, output, actions
            
        except Exception as e:
            self.logger.error(f"完整动作生成流程失败: {str(e)}")
            # 返回错误情况下的默认响应
            error_prompt = self.generate_prompt(account, market) if account and market else "Error generating prompt"
            error_output = json.dumps({
                "reasoning": f"动作生成失败: {str(e)}",
                "action_type": "NONE"
            }, ensure_ascii=False)
            error_actions = [NoneActionSchema(
                reasoning=f"动作生成失败: {str(e)}",
                action_type=ActionType.NONE
            )]
            
            return error_prompt, error_output, error_actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_device": self.model.device if self.model else None
        }


class ActionParseError(Exception):
    """动作解析异常"""
    pass