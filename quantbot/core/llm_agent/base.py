# llm_agent.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List
from datetime import datetime
import logging

from general.interfaces.llm_agent_interface import LLMAgentInterface
from config.llm_agent_config import AgentConfig
from general.types.action_types import ActionSpace, AddOrderAction, CancelOrderAction, NoneAction
from general.types.order_types import OrderType, OrderStatus, OrderInterface
from general.types.obervation_types import ObervationSpace

class LLMAgent(LLMAgentInterface):
    """LLM Agent 实现类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化LLM Agent"""
        self.config = AgentConfig()
        if config:
            self._update_config(config)
            
        self.logger = logging.getLogger(__name__)
        self._load_model()
        self._initialize_conversation_template()
        
    def _update_config(self, config: Dict[str, Any]):
        """更新配置参数"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif isinstance(self.config.model_config, dict) and key in self.config.model_config:
                self.config.model_config[key] = value
    
    def _load_model(self):
        """从本地加载模型和tokenizer"""
        try:
            self.logger.info(f"加载模型从: {self.config.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 设置模型推理参数
            self.generation_config = {
                "max_new_tokens": 1024,
                "temperature": self.config.model_config["temperature"],
                "top_p": self.config.model_config["top_p"],
                "do_sample": self.config.model_config["do_sample"],
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            self.logger.info("模型加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _initialize_conversation_template(self):
        """初始化对话模板"""
        self.conversation_template = {
            "system": self.config.system_prompt,
            "user": "当前市场状态：\n{observation_json}\n\n请基于以上信息进行分析并生成交易决策："
        }
    
    def _observation_to_json(self, observation: ObervationSpace) -> str:
        """将observation对象转换为JSON字符串"""
        try:
            # 转换为字典
            obs_dict = observation.dict()
            
            # 处理特殊字段
            if 'timestamp' in obs_dict:
                obs_dict['timestamp'] = obs_dict['timestamp'].isoformat()

            
            return json.dumps(obs_dict, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Observation转换JSON失败: {e}")
            return "{}"
    
    def _prepare_model_input(self, observation_json: str) -> str:
        """准备模型输入"""
        user_prompt = self.conversation_template["user"].format(
            observation_json=observation_json
        )
        
        full_prompt = f"System: {self.conversation_template['system']}\n\nUser: {user_prompt}\n\nAssistant:"
        return full_prompt
    
    def _generate_model_output(self, prompt: str) -> str:
        """模型推理生成输出"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids.to(self.model.device),
                    **self.generation_config
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取模型新增的回复部分
            assistant_response = response.split("Assistant:")[-1].strip()
            
            return assistant_response
            
        except Exception as e:
            self.logger.error(f"模型推理失败: {e}")
            return '{"reasoning": "模型推理失败", "action_type": "NONE"}'
    
    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        """解析模型输出"""
        try:
            # 提取JSON部分
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("未找到有效的JSON输出")
            
            json_str = output[start_idx:end_idx]
            parsed_output = json.loads(json_str)
            
            # 验证必需字段
            if "reasoning" not in parsed_output:
                parsed_output["reasoning"] = "推理过程未提供"
            if "action_type" not in parsed_output:
                parsed_output["action_type"] = "NONE"
            
            return parsed_output
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"解析模型输出失败: {e}, 使用默认无操作")
            return {
                "reasoning": f"解析失败: {str(e)}",
                "action_type": "NONE"
            }
    
    def _create_action(self, parsed_output: Dict[str, Any]) -> ActionSpace:
        """根据解析结果创建Action"""
        reasoning = parsed_output.get("reasoning", "")
        action_type = parsed_output.get("action_type", "NONE")
        
        if action_type == "ADD_ORDER":
            order_details = parsed_output.get("order_details", {})
            
            # 创建Order对象
            order_interface = OrderInterface(
                symbol=order_details.get("symbol", ""),
                order_type=OrderType(order_details.get("order_type", "BUY")),
                price=float(order_details.get("price", 0)),
                quantity=int(order_details.get("quantity", 0)),
            )
            
            return AddOrderAction(
                reasoning=reasoning,
                order_interface=order_interface
            )
            
        elif action_type == "CANCEL_ORDER":
            order_details = parsed_output.get("order_details", {})
            return CancelOrderAction(
                reasoning=reasoning,
                order_id=order_details.get("order_id", "")
            )
        
        else:  # NONE 或其他未知类型
            return NoneAction(
                reasoning=reasoning
            )
    
    def generate_action(self, observation: ObervationSpace) -> Dict[str, Any]:
        """基于状态观察生成动作"""
        try:
            # 第一步：准备系统提示词
            self.logger.info("准备系统提示词和对话模板")
            
            # 第二步：转换observation为JSON
            self.logger.info("转换observation为JSON格式")
            observation_json = self._observation_to_json(observation)
            
            # 第三步：准备模型输入
            prompt = self._prepare_model_input(observation_json)
            
            # 第四步：模型推理
            self.logger.info("开始模型推理")
            model_output = self._generate_model_output(prompt)
            
            # 第五步：解析输出
            self.logger.info("解析模型输出")
            parsed_output = self._parse_model_output(model_output)
            
            # 第六步：创建Action
            self.logger.info("创建Action对象")
            action = self._create_action(parsed_output)
            
            self.logger.info(f"决策完成: {action.action_type}")
            return action.dict()
            
        except Exception as e:
            self.logger.error(f"生成动作失败: {e}")
            # 返回无操作作为降级方案
            none_action = NoneAction(
                reasoning=f"决策过程异常: {str(e)}"
            )
            return none_action.dict()