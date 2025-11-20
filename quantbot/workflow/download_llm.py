"""
下载 LLM 模型到本地
"""
import os
import sys
from pathlib import Path
import logging

# 添加项目路径
script_dir = Path(__file__).parent.resolve()
quantbot_dir = script_dir.parent.resolve()
sys.path.insert(0, str(quantbot_dir))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_name: str = 'deepseek-ai/DeepSeek-V3', 
                   local_dir: str = None):
    """
    下载模型到本地目录
    
    Args:
        model_name: HuggingFace 模型名称
        local_dir: 本地保存目录，如果为 None 则使用 quantbot/llm/{model_name}
    """
    try:
        # 确定本地保存目录
        if local_dir is None:
            # 从模型名称提取目录名（去掉 deepseek-ai/ 前缀）
            model_dir_name = model_name.split('/')[-1]
            local_dir = quantbot_dir / 'llm' / model_dir_name
        else:
            local_dir = Path(local_dir)
        
        local_dir = local_dir.resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始下载模型: {model_name}")
        logger.info(f"保存目录: {local_dir}")
        
        # 检测 GPU 可用性
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("未检测到 GPU，将使用 CPU 模式")
        
        # 下载分词器
        logger.info("正在下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(local_dir))
        logger.info("✓ 分词器下载完成")
        
        # 下载模型（这可能需要较长时间）
        logger.info("正在下载模型（这可能需要较长时间，请耐心等待）...")
        logger.info("模型文件较大，请确保网络连接稳定...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(str(local_dir))
        logger.info("✓ 模型下载完成")
        
        logger.info(f"\n模型已成功下载到: {local_dir}")
        logger.info(f"使用本地模型时，请将 model_path 设置为: {local_dir}")
        
        return str(local_dir)
        
    except Exception as e:
        logger.error(f"下载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载 DeepSeek-V3 模型')
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-ai/DeepSeek-V3',
        help='要下载的模型名称（默认: deepseek-ai/DeepSeek-V3）'
    )
    parser.add_argument(
        '--local-dir',
        type=str,
        default=None,
        help='本地保存目录（默认: quantbot/llm/{model_name}）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek-V3 模型下载工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    if args.local_dir:
        print(f"保存目录: {args.local_dir}")
    else:
        print(f"保存目录: quantbot/llm/{args.model.split('/')[-1]}")
    print("=" * 60)
    print()
    
    try:
        download_model(args.model, args.local_dir)
        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

