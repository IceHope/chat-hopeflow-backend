import os
from typing import Optional, List

from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks import CallbackManager
from llama_index.core.prompts import PromptType
from llama_index.core.schema import NodeWithScore

from utils.log_utils import LogUtils

RAG_QUERY_PROMPT = """
你是一个专门用于问答任务的智能助手。
请仅使用以下提供的上下文信息来回答问题。
如果你不知道答案,请直接说你不知道。
只能根据提供的内容回答问题,不要使用任何其他知识来源,包括你自身的训练数据。
不要回答与问题无关的内容,不要臆测或假设答案。
尽可能简洁地回答问题。
始终使用简体中文回答。
仅基于给定的上下文信息回答,不要使用先验知识。


问题: {query_str}
上下文: {context_str}
回答:
"""

RAG_TEXT_QA_PROMPT = PromptTemplate(
    RAG_QUERY_PROMPT, prompt_type=PromptType.QUESTION_ANSWER
)


class GenerateManager:
    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager or Settings.callback_manager
        from llama_index.llms.openai import OpenAI

        generate_llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPEN_AGI_API_KEY"),
            api_base=os.getenv("OPEN_AGI_BASE_URL"),
            temperature=0.7,
        )

        self.response_synthesizer = get_response_synthesizer(
            text_qa_template=RAG_TEXT_QA_PROMPT,
            llm=generate_llm,
            callback_manager=self.callback_manager,
            streaming=True,
        )

    def generate_stream_response(
        self, query_str: str, nodes: List[NodeWithScore]
    ) -> RESPONSE_TYPE:
        response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=nodes,
        )

        return response
