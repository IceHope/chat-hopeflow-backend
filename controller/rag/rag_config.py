from typing import Optional
from pydantic import BaseModel, Field


class RagFrontendConfig(BaseModel):
    rag_rerank_count: Optional[int] = Field(default=3, description="重排序后的文本数量")
    rag_retrieve_count: Optional[int] = Field(default=6, description="检索的文本数量")
    rag_fusion_count: Optional[int] = Field(default=3, description="类似语义的查询数量")
