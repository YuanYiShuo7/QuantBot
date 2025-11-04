from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..schemas.account_schema import AccountSchema
from ..schemas.order_schema import OrderSchema, OrderFormSchema, OrderResultSchema

class AccountInterface(ABC):
    """账户接口抽象类 - 定义账户相关操作的标准接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化账户接口
        
        Args:
            config: 配置字典，包含账户相关配置参数
        """
        pass

    @abstractmethod
    def format_account_info_for_prompt(self) -> str:
        """
        将账户信息 AccountSchema 转换为 Agent 的 Prompt 文本
        
        Returns:
            str: 格式化的账户信息文本
        """
        pass