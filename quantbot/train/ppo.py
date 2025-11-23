import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import json
import logging
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionSimilarityCalculator:
    """计算动作相似度的工具类"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    def parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """从文本中解析动作"""
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个文本
                json_str = text
            
            # 解析JSON
            actions = json.loads(json_str)
            
            # 统一格式为列表
            if isinstance(actions, dict):
                actions = [actions]
                
            return actions
            
        except Exception as e:
            logger.warning(f"解析动作失败: {e}, 文本: {text[:100]}...")
            return []
    
    def action_to_text(self, action: Dict[str, Any]) -> str:
        """将动作转换为文本表示用于相似度计算"""
        parts = []
        
        # 添加动作类型
        action_type = action.get('action_type', '')
        parts.append(f"action_type:{action_type}")
        
        # 添加推理（如果有）
        reasoning = action.get('reasoning', '')
        if reasoning:
            parts.append(f"reasoning:{reasoning}")
        
        # 根据动作类型添加特定字段
        if action_type == 'ADD_ORDER':
            parts.extend([
                f"symbol:{action.get('symbol', '')}",
                f"order_type:{action.get('order_type', '')}",
                f"price:{action.get('price', 0)}",
                f"quantity:{action.get('quantity', 0)}"
            ])
        elif action_type == 'CANCEL_ORDER':
            parts.append(f"order_id:{action.get('order_id', '')}")
        elif action_type == 'NONE':
            parts.append("no_action")
        
        return " ".join(parts)
    
    def calculate_similarity(self, response_actions: List[Dict], output_actions: List[Dict]) -> float:
        """计算两组动作之间的余弦相似度"""
        if not response_actions and not output_actions:
            return 1.0  # 两者都为空，完全相似
        
        if not response_actions or not output_actions:
            return 0.0  # 一个为空一个不为空，完全不相似
        
        # 将动作转换为文本
        response_texts = [self.action_to_text(action) for action in response_actions]
        output_texts = [self.action_to_text(action) for action in output_actions]
        
        # 合并所有文本来拟合TF-IDF向量化器
        all_texts = response_texts + output_texts
        
        try:
            # 计算TF-IDF向量
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # 分离响应和输出的向量
            response_vectors = tfidf_matrix[:len(response_texts)]
            output_vectors = tfidf_matrix[len(response_texts):]
            
            # 计算平均相似度
            similarities = []
            for resp_vec in response_vectors:
                for out_vec in output_vectors:
                    sim = cosine_similarity(resp_vec, out_vec)[0][0]
                    similarities.append(sim)
            
            if similarities:
                return float(np.mean(similarities))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"计算相似度失败: {e}")
            return 0.0

class PPOTrainer:
    def __init__(
        self,
        model_path: str = "quantbot/llm/Qwen2-7B-Instruct",
        lora_checkpoint_path: str = "quantbot/checkpoint/Qwen2-7B-Instruct",
        trajectory_path: str = "quantbot/trajectory/trajectories.json",
        output_dir: str = "quantbot/checkpoint/Qwen2-7B-Instruct-lora-ppo",
        config: Dict[str, Any] = None
    ):
        self.model_path = model_path
        self.lora_checkpoint_path = lora_checkpoint_path
        self.trajectory_path = trajectory_path
        self.output_dir = output_dir
        self.config = config or {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化相似度计算器
        self.similarity_calculator = ActionSimilarityCalculator()
        
        # 初始化模型和tokenizer
        self.model, self.ref_model, self.tokenizer = self._load_models()
        
    def _load_models(self) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
        """加载模型、参考模型和tokenizer"""
        logger.info(f"从 {self.model_path} 加载模型...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 模型配置
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
            "use_cache": False,  # PPO训练需要
        }
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        # 检查是否有现有的LoRA检查点
        if os.path.exists(self.lora_checkpoint_path) and any(
            f.startswith("adapter") for f in os.listdir(self.lora_checkpoint_path)
        ):
            logger.info(f"发现LoRA检查点，从 {self.lora_checkpoint_path} 加载...")
            model = PeftModel.from_pretrained(
                base_model, 
                self.lora_checkpoint_path,
                torch_dtype=torch.float16
            )
            ref_model = PeftModel.from_pretrained(
                base_model,
                self.lora_checkpoint_path,
                torch_dtype=torch.float16
            )
        else:
            logger.info("未发现LoRA检查点，初始化新的LoRA训练...")
            
            # 配置LoRA
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # 应用LoRA
            model = get_peft_model(base_model, lora_config)
            ref_model = get_peft_model(base_model, lora_config)
        
        # 准备模型用于训练
        model = prepare_model_for_kbit_training(model)
        ref_model = prepare_model_for_kbit_training(ref_model)
        
        # 设置参考模型为评估模式
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
            
        logger.info(f"可训练参数数量: {model.print_trainable_parameters()}")
        return model, ref_model, tokenizer
    
    def _load_trajectory_data(self) -> List[Dict[str, Any]]:
        """加载轨迹数据并转换为PPO训练格式"""
        logger.info(f"从 {self.trajectory_path} 加载轨迹数据...")
        
        try:
            with open(self.trajectory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取训练样本
            samples = []
            
            def extract_samples(obj, path=""):
                """递归提取样本数据"""
                if isinstance(obj, dict):
                    # 检查是否包含所需的字段
                    if all(key in obj for key in ['prompt', 'output', 'score']):
                        samples.append(obj)
                    # 递归检查所有值
                    for key, value in obj.items():
                        extract_samples(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for item in obj:
                        extract_samples(item, path)
            
            extract_samples(data)
            
            logger.info(f"加载了 {len(samples)} 个训练样本")
            return samples
            
        except Exception as e:
            logger.error(f"加载轨迹数据失败: {str(e)}")
            return []
    
    def _prepare_ppo_dataset(self, samples: List[Dict[str, Any]]) -> Dataset:
        """准备PPO训练数据集"""
        logger.info("准备PPO训练数据...")
        
        queries = []
        original_outputs = []
        scores = []
        
        for sample in samples:
            prompt = sample['prompt']
            output = sample['output']
            score = sample['score'] or 0.0
            
            queries.append(prompt)
            original_outputs.append(output)
            scores.append(score)
        
        # 创建数据集
        dataset_dict = {
            "query": queries,
            "original_output": original_outputs,
            "score": scores,
        }
        
        # 统计分数分布
        scores_array = np.array(scores)
        logger.info(f"分数统计 - 均值: {np.mean(scores_array):.4f}, "
                   f"标准差: {np.std(scores_array):.4f}, "
                   f"最小值: {np.min(scores_array):.4f}, "
                   f"最大值: {np.max(scores_array):.4f}")
        
        return Dataset.from_dict(dataset_dict)
    
    def _format_prompt(self, prompt: str) -> str:
        """格式化prompt为模型输入格式"""
        # 使用Qwen的对话格式
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted_text
    
    def _calculate_reward(self, response: str, original_output: str, score: float) -> float:
        """计算奖励: reward = score * similarity"""
        try:
            # 解析动作
            response_actions = self.similarity_calculator.parse_actions(response)
            original_actions = self.similarity_calculator.parse_actions(original_output)
            
            # 计算相似度
            similarity = self.similarity_calculator.calculate_similarity(
                response_actions, original_actions
            )
            
            # 计算最终奖励
            reward = score * similarity
            
            logger.debug(f"奖励计算: score={score:.4f}, similarity={similarity:.4f}, reward={reward:.4f}")
            return reward
            
        except Exception as e:
            logger.warning(f"计算奖励失败: {e}, 使用默认奖励")
            return score  # 失败时回退到原始分数
    
    def train_ppo(
        self,
        num_epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 1.41e-5,
        clip_range: float = 0.2,
        kl_penalty: str = "kl",
        target_kl: float = 6.0,
        min_similarity_threshold: float = 0.0  # 相似度阈值，低于此值则惩罚
    ):
        """使用PPO进行强化学习训练，奖励 = score * similarity"""
        logger.info("开始PPO强化学习训练...")
        
        # 加载数据
        samples = self._load_trajectory_data()
        if not samples:
            logger.error("没有找到有效的训练样本")
            return
        
        dataset = self._prepare_ppo_dataset(samples)
        
        # PPO配置
        ppo_config = PPOConfig(
            model_name=self.model_path,
            learning_rate=learning_rate,
            ppo_epochs=num_epochs,
            batch_size=batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=4,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=target_kl,
            kl_penalty=kl_penalty,
            cliprange=clip_range,
            cliprange_value=clip_range,
            vf_coef=0.1,
            seed=42,
            log_with=None,
            project_kwargs={"logging_dir": self.output_dir},
        )
        
        # 创建PPO训练器
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
        )
        
        # 训练循环
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 512,  # 增加生成长度以包含完整JSON
        }
        
        logger.info("开始PPO训练循环...")
        
        for epoch in range(num_epochs):
            logger.info(f"开始第 {epoch + 1}/{num_epochs} 轮训练")
            
            total_similarity = 0
            total_reward = 0
            batch_count = 0
            
            for batch in ppo_trainer.dataloader:
                # 获取查询、原始输出和分数
                queries = batch["query"]
                original_outputs = batch["original_output"]
                scores = batch["score"]
                
                # 获取查询的token
                query_tensors = []
                for query in queries:
                    formatted_query = self._format_prompt(query)
                    query_tensor = self.tokenizer.encode(formatted_query, return_tensors="pt").squeeze()
                    query_tensors.append(query_tensor)
                
                # 生成响应
                response_tensors = []
                for query_tensor in query_tensors:
                    response = ppo_trainer.generate(
                        query_tensor.unsqueeze(0).to(self.device),
                        **generation_kwargs
                    )
                    response_tensors.append(response.squeeze())
                
                # 解码响应
                responses = [
                    self.tokenizer.decode(r, skip_special_tokens=True) 
                    for r in response_tensors
                ]
                
                # 计算每个响应的奖励
                reward_tensors = []
                batch_similarities = []
                
                for i, (response, original_output, score) in enumerate(zip(responses, original_outputs, scores)):
                    reward = self._calculate_reward(response, original_output, score)
                    reward_tensors.append(torch.tensor(reward))
                    
                    # 记录相似度统计
                    response_actions = self.similarity_calculator.parse_actions(response)
                    original_actions = self.similarity_calculator.parse_actions(original_output)
                    similarity = self.similarity_calculator.calculate_similarity(response_actions, original_actions)
                    batch_similarities.append(similarity)
                    
                    # 记录前几个样本的详细信息
                    if i < 2:  # 只记录前2个样本的详细信息
                        logger.info(f"样本 {i+1}:")
                        logger.info(f"  原始输出: {original_output[:100]}...")
                        logger.info(f"  生成响应: {response[:100]}...")
                        logger.info(f"  分数: {score:.4f}, 相似度: {similarity:.4f}, 奖励: {reward:.4f}")
                
                # PPO训练步骤
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                
                # 更新统计
                batch_avg_similarity = np.mean(batch_similarities)
                batch_avg_reward = np.mean([r.item() for r in reward_tensors])
                total_similarity += batch_avg_similarity
                total_reward += batch_avg_reward
                batch_count += 1
                
                # 记录训练统计
                if stats is not None and batch_count % 5 == 0:
                    avg_similarity = total_similarity / batch_count
                    avg_reward = total_reward / batch_count
                    logger.info(f"PPO步骤 {batch_count}:")
                    logger.info(f"  平均相似度: {avg_similarity:.4f}")
                    logger.info(f"  平均奖励: {avg_reward:.4f}")
                    logger.info(f"  PPO统计: {stats}")
            
            # 每轮结束后保存检查点
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
            ppo_trainer.save_pretrained(checkpoint_dir)
            
            # 输出本轮统计
            if batch_count > 0:
                final_avg_similarity = total_similarity / batch_count
                final_avg_reward = total_reward / batch_count
                logger.info(f"第 {epoch+1} 轮结束 - 平均相似度: {final_avg_similarity:.4f}, 平均奖励: {final_avg_reward:.4f}")
            
            logger.info(f"检查点已保存到: {checkpoint_dir}")
        
        # 保存最终模型
        final_dir = os.path.join(self.output_dir, "final")
        ppo_trainer.save_pretrained(final_dir)
        logger.info(f"最终模型已保存到: {final_dir}")
        
        # 保存最终模型
        final_dir = os.path.join(self.output_dir, "final")
        ppo_trainer.save_pretrained(final_dir)
        logger.info(f"最终模型已保存到: {final_dir}")

def main():
    """主训练函数"""
    # 配置参数
    config = {
        "model_path": "quantbot/llm/Qwen2-7B-Instruct",
        "lora_checkpoint_path": "quantbot/checkpoint/Qwen2-7B-Instruct",
        "trajectory_path": "quantbot/trajectory/trajectories.json",
        "output_dir": "quantbot/checkpoint/Qwen2-7B-Instruct-lora-ppo",
    }
    
    # 创建训练器
    trainer = PPOTrainer(**config)
    
    # 运行PPO训练
    try:
        # 方法1: 标准PPO训练（使用score * similarity作为奖励）
        trainer.train_ppo(
            num_epochs=1,
            batch_size=2,
            learning_rate=1.41e-5
        )
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()