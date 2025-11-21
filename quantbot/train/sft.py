import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import logging
from typing import Dict, List, Any
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoreWeightedTrainer:
    def __init__(
        self,
        model_path: str = "quantbot/llm/Qwen2-7B-Instruct",
        lora_checkpoint_path: str = "quantbot/checkpoint/Qwen2-7B-Instruct",
        trajectory_path: str = "quantbot/trajectory/id_trajectory.json",
        output_dir: str = "quantbot/checkpoint/Qwen2-7B-Instruct-lora"
    ):
        self.model_path = model_path
        self.lora_checkpoint_path = lora_checkpoint_path
        self.trajectory_path = trajectory_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型和tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        logger.info(f"从 {self.model_path} 加载模型...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 检查是否有现有的LoRA检查点
        if os.path.exists(self.lora_checkpoint_path) and any(
            f.startswith("adapter") for f in os.listdir(self.lora_checkpoint_path)
        ):
            logger.info(f"发现LoRA检查点，从 {self.lora_checkpoint_path} 加载...")
            
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            model = PeftModel.from_pretrained(
                model, 
                self.lora_checkpoint_path,
                torch_dtype=torch.float16
            )
            
        else:
            logger.info("未发现LoRA检查点，初始化新的LoRA训练...")
            
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 配置LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # LoRA秩
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # 应用LoRA
            model = get_peft_model(model, lora_config)
        
        # 准备模型用于训练
        model = prepare_model_for_kbit_training(model)
        
        logger.info(f"可训练参数数量: {model.print_trainable_parameters()}")
        return model, tokenizer
    
    def _load_trajectory_data(self) -> List[Dict[str, Any]]:
        """加载轨迹数据"""
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
    
    def _calculate_sample_weights(self, scores: List[float]) -> List[float]:
        """负值降低概率，正值提高概率"""
        scores_array = np.array(scores)
        weights = []
        
        for score in scores_array:
            if score > 0:
                # 正分：提高概率，权重 > 1
                weight = 1 + abs(score) * 2  # 例如 score=0.5 -> weight=2.0
            elif score < 0:
                # 负分：降低概率，权重 < 1  
                weight = 1 / (1 + abs(score) * 2)  # 例如 score=-0.5 -> weight=0.5
            else:
                # 零分：保持原权重
                weight = 1.0
            weights.append(weight)
        
        weights = np.array(weights)
        
        # 输出详细统计
        pos_scores = scores_array[scores_array > 0]
        neg_scores = scores_array[scores_array < 0]
        
        if len(pos_scores) > 0:
            logger.info(f"正分样本: {len(pos_scores)}个, 平均分: {np.mean(pos_scores):.4f}, 平均权重: {np.mean(weights[scores_array > 0]):.4f}")
        if len(neg_scores) > 0:
            logger.info(f"负分样本: {len(neg_scores)}个, 平均分: {np.mean(neg_scores):.4f}, 平均权重: {np.mean(weights[scores_array < 0]):.4f}")
        
        return weights.tolist()
    
    def _prepare_training_data(self, samples: List[Dict[str, Any]]) -> Dataset:
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        prompts = []
        responses = []
        scores = []
        
        for sample in samples:
            prompt = sample['prompt']
            output = sample['output']
            score = sample['score']
            
            # 从output中提取JSON部分作为响应
            if '```json' in output:
                # 提取JSON内容
                start_idx = output.find('```json') + 7
                end_idx = output.find('```', start_idx)
                if end_idx != -1:
                    response = output[start_idx:end_idx].strip()
                else:
                    response = output
            else:
                response = output
            
            # 清理响应中的多余内容
            response = response.replace('\n', ' ').strip()
            
            prompts.append(prompt)
            responses.append(response)
            scores.append(score)
        
        # 计算样本权重
        sample_weights = self._calculate_sample_weights(scores)
        
        # 创建数据集
        dataset_dict = {
            "prompt": prompts,
            "response": responses,
            "score": scores,
            "weight": sample_weights
        }
        
        logger.info(f"数据准备完成，正样本数: {sum(1 for s in scores if s > 0)}")
        logger.info(f"负样本数: {sum(1 for s in scores if s < 0)}")
        
        return Dataset.from_dict(dataset_dict)
    
    def _format_prompt_response(self, prompt: str, response: str) -> str:
        """格式化prompt和response为模型输入格式"""
        # 使用Qwen的对话格式
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        return formatted_text
    
    def train_with_weighted_sft(self, num_epochs: int = 5, batch_size: int = 2):
        """使用加权监督微调进行训练"""
        logger.info("开始加权监督微调训练...")
        
        # 加载数据
        samples = self._load_trajectory_data()
        if not samples:
            logger.error("没有找到有效的训练样本")
            return
        
        dataset = self._prepare_training_data(samples)
        
        def preprocess_function(examples):
            """预处理函数"""
            texts = []
            weights = []
            
            for prompt, response, weight in zip(examples['prompt'], examples['response'], examples['weight']):
                formatted_text = self._format_prompt_response(prompt, response)
                texts.append(formatted_text)
                weights.append(weight)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_tensors=None
            )
            
            # 对于因果语言建模，labels和input_ids相同
            tokenized["labels"] = tokenized["input_ids"].copy()
            tokenized["weights"] = weights
            
            return tokenized
        
        # 预处理数据集
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 自定义加权训练器
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                """
                重写损失计算函数，加入样本权重
                """
                # 提取权重
                weights = inputs.pop("weights")
                
                # 前向传播
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # 计算加权损失
                labels = inputs.get("labels")
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())
                
                # 应用序列级别的权重
                seq_weights = torch.tensor(weights, device=loss.device).unsqueeze(1).expand_as(loss)
                weighted_loss = (loss * seq_weights).mean()
                
                return (weighted_loss, outputs) if return_outputs else weighted_loss
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            learning_rate=1e-4,
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to=None,  # 不使用wandb等
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info("开始训练...")
        train_result = trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 记录训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"训练完成，模型保存到: {self.output_dir}")
        logger.info(f"训练损失: {metrics.get('train_loss', 'N/A')}")
    
    def train_with_filtered_sft(self, num_epochs: int = 5, batch_size: int = 2, score_threshold: float = 0.0):
        """使用基于score过滤的监督微调"""
        logger.info(f"开始基于score过滤的监督微调训练，阈值: {score_threshold}")
        
        # 加载数据
        samples = self._load_trajectory_data()
        if not samples:
            logger.error("没有找到有效的训练样本")
            return
        
        # 过滤样本
        filtered_samples = [s for s in samples if s['score'] >= score_threshold]
        logger.info(f"过滤后样本数: {len(filtered_samples)}/{len(samples)}")
        
        if not filtered_samples:
            logger.warning("没有样本通过过滤，使用所有样本")
            filtered_samples = samples
        
        dataset = self._prepare_training_data(filtered_samples)
        
        def preprocess_function(examples):
            """预处理函数"""
            texts = []
            for prompt, response in zip(examples['prompt'], examples['response']):
                formatted_text = self._format_prompt_response(prompt, response)
                texts.append(formatted_text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_tensors=None
            )
            
            # 对于因果语言建模，labels和input_ids相同
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # 预处理数据集
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=200,
            learning_rate=1e-4,
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=True,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"训练完成，模型保存到: {self.output_dir}")

def main():
    """主函数"""
    # 初始化训练器
    trainer = ScoreWeightedTrainer()
    
    # 选择训练方法
    method = "weighted"  # 可选: "weighted", "filtered"
    
    if method == "weighted":
        # 使用加权监督微调
        trainer.train_with_weighted_sft(num_epochs=5, batch_size=2)
    elif method == "filtered":
        # 使用基于score过滤的监督微调
        trainer.train_with_filtered_sft(num_epochs=5, batch_size=2, score_threshold=0.0)
    else:
        logger.error(f"未知的训练方法: {method}")

if __name__ == "__main__":
    main()