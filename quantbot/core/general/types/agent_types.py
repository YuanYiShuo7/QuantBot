from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
from action_types import ActionSpace

class Response(BaseModel):
    """动作执行响应"""
    action: ActionSpace = Field(..., description="执行的动作")
    success: bool = Field(..., description="是否执行成功")
    executed_price: Optional[float] = Field(None, description="实际成交价格")
    executed_quantity: Optional[int] = Field(None, description="实际成交数量")
    message: str = Field(..., description="执行结果消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="执行时间戳")

class Output(BaseModel):
    """LLM动作输出格式"""
    reasoning: str = Field(..., description="决策推理过程")
    action: ActionSpace = Field(..., description="具体交易动作")