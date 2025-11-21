"""
下载 LLM 模型到本地（使用 ModelScope）
"""
import os
import sys
from pathlib import Path
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    logger.error(f"导入依赖失败: {e}")
    logger.info("请安装: pip install modelscope transformers torch")
    sys.exit(1)


def check_disk_space(required_gb: int = 20):
    """检查磁盘空间是否足够"""
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        logger.info(f"当前磁盘可用空间: {free_gb}GB")
        
        if free_gb < required_gb:
            logger.warning(f"磁盘空间可能不足！需要约{required_gb}GB，当前仅有{free_gb}GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"无法检查磁盘空间: {str(e)}")
        return True


def download_model(model_name: str = 'Qwen/Qwen2-7B-Instruct', 
                   local_dir: str = None,
                   cache_dir: str = None):
    """
    使用 ModelScope 下载模型到本地目录
    
    Args:
        model_name: 模型名称
        local_dir: 本地保存目录
        cache_dir: 缓存目录
    """
    try:
        # 设置默认缓存目录
        if cache_dir is None:
            cache_dir = "./quantbot/cache/model_cache"
        
        # 设置缓存目录
        cache_path = Path(cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ['MODELSCOPE_CACHE'] = str(cache_path)
        
        # 确定本地保存目录
        if local_dir is None:
            model_dir_name = model_name.split('/')[-1]
            local_dir = Path("./quantbot/llm") / model_dir_name
        else:
            local_dir = Path(local_dir)
        
        local_dir = local_dir.resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始下载模型: {model_name}")
        logger.info(f"保存目录: {local_dir}")
        logger.info(f"缓存目录: {cache_path}")
        
        # 检查磁盘空间
        check_disk_space(20)
        
        # 使用 ModelScope 下载模型
        logger.info("正在下载模型（使用 ModelScope）...")
        
        downloaded_path = snapshot_download(
            model_id=model_name,
            cache_dir=str(cache_path),
            local_dir=str(local_dir),
            revision='master'
        )
        
        logger.info("✓ 模型下载完成")
        
        # 验证模型文件
        logger.info("验证模型文件...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_dir),
                trust_remote_code=True
            )
            logger.info("✓ 分词器加载成功")
            
            model = AutoModelForCausalLM.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("✓ 模型加载成功")
            
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数量: {total_params / 1e9:.1f}B")
            
        except Exception as e:
            logger.warning(f"模型验证警告: {str(e)}")
        
        logger.info(f"🎉 模型已成功下载到: {local_dir}")
        return str(local_dir)
        
    except Exception as e:
        logger.error(f"下载模型失败: {str(e)}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 ModelScope 下载模型')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2-1.5B-Instruct',
        help='要下载的模型名称'
    )
    parser.add_argument(
        '--local-dir',
        type=str,
        default=None,
        help='本地保存目录（默认: ./quantbot/llm/{模型名}）'
    )
    parser.add_argument(
        '--cache-dir', 
        type=str,
        default=None,
        help='缓存目录（默认: ./quantbot/cache/model_cache）'
    )
    
    args = parser.parse_args()
    
    # 构建完整的保存路径
    if args.local_dir is None:
        model_dir_name = args.model.split('/')[-1]
        full_local_dir = f"./quantbot/llm/{model_dir_name}"
    else:
        full_local_dir = args.local_dir
    
    # 构建缓存路径
    if args.cache_dir is None:
        full_cache_dir = "./quantbot/cache/model_cache"
    else:
        full_cache_dir = args.cache_dir
    
    print("=" * 50)
    print("ModelScope 模型下载工具")
    print("=" * 50)
    print(f"模型: {args.model}")
    print(f"保存目录: {full_local_dir}")
    print(f"缓存目录: {full_cache_dir}")
    print("=" * 50)
    
    try:
        download_model(args.model, args.local_dir, args.cache_dir)
        print("\n🎉 下载完成！")
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()