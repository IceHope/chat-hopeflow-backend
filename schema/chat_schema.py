from typing import Optional, List

from pydantic import BaseModel, Field


class ChatRequestData(BaseModel):
    multi_turn_chat_enabled: bool = Field(default=True, description="是否多轮对话")
    user_name: str = Field(description="用户名")
    session_id: int = Field(description="会话id,时间戳")
    data: str = Field(description="用户的问题")
    command: Optional[str] = Field(default=None, description="通用指令")
    model_type: Optional[str] = Field(default=None, description="模型类型")
    model_name: Optional[str] = Field(default=None, description="模型名称")
    image_urls: Optional[List[str]] = Field(default=None, description="图片地址")
