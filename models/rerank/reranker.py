from typing import Optional

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.llms import LLM
import os

from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.postprocessor.jinaai_rerank.base import JinaRerank

load_dotenv()


class RagReranker:
    """重排序3种类型
    1.排序模型
    2.jina
    3.大模型排序
    """

    @staticmethod
    def get_transformer_cross_encoder(
        model_path: str = None,
        top_n: int = 3,
    ) -> SentenceTransformerRerank:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        # 本地模型
        if model_path is None:
            model_path = os.getenv("RERANK_BGE_LARGE")
        reranker = SentenceTransformerRerank(model=model_path, top_n=top_n)
        return reranker

    @staticmethod
    def get_jina_rerank(top_n: int = 3) -> JinaRerank:
        # pip install llama-index-postprocessor-jinaai-rerank
        from llama_index.postprocessor.jinaai_rerank import JinaRerank

        # jina-reranker-v1-base-en
        # jina-reranker-v2-base-multilingual
        # 远程模型
        jina_reranker = JinaRerank(
            top_n=top_n,
            model="jina-reranker-v1-base-en",
            api_key=os.getenv("JINA_API_KEY"),
        )
        return jina_reranker

    @staticmethod
    def get_llm_rerank(
        llm: Optional[LLM] = Settings.llm, top_n: int = 3
    ) -> RankGPTRerank:
        from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank

        llm_rerank = RankGPTRerank(llm=llm, top_n=top_n)
        return llm_rerank
