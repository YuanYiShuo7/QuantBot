# LLM 模型目录

此目录用于存储本地下载的 LLM（大语言模型）文件。

## 目录说明

- 模型文件通常较大（几十GB），因此此目录默认被 `.gitignore` 忽略
- 下载的模型会保存在以模型名称命名的子目录中
- 例如：`DeepSeek-V3/` 目录包含 DeepSeek-V3 模型的所有文件

## 使用方法

### 下载模型

使用项目提供的下载脚本：

```bash
python quantbot/workflow/download_llm.py
```

或者指定模型和保存路径：

```bash
python quantbot/workflow/download_llm.py --model deepseek-ai/DeepSeek-V3 --local-dir ./quantbot/llm/DeepSeek-V3
```

### 在配置中使用本地模型

在 Simulator 或 LLMAgent 配置中，将 `model_path` 设置为本地路径：

```python
{
    'llm_agent': {
        'model_path': './quantbot/llm/DeepSeek-V3',  # 或使用绝对路径
        'device': 'cuda',  # 或 'cpu'
        'temperature': 0.7,
        'max_new_tokens': 1024,
    }
}
```

## 注意事项

- 模型文件较大，确保有足够的磁盘空间
- 首次下载需要稳定的网络连接
- 如果使用 CPU 模式，请添加 `--use-cpu` 参数
- 模型文件不会被 Git 跟踪，需要手动管理

