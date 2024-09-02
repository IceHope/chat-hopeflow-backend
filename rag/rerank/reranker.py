from typing import Optional

from llama_index.core import Settings
from llama_index.core.llms import LLM


class RagReranker:
    """重排序3种类型
    1.排序模型
    2.jina
    3.大模型排序
    """

    @staticmethod
    def get_transformer_cross_encoder(top_n: int = 3):
        from llama_index.core.postprocessor import SentenceTransformerRerank
        # 本地模型
        path = "F:/HuggingFace/Rerank/bce-reranker-base_v1"
        reranker = SentenceTransformerRerank(model=path, top_n=top_n)
        return reranker

    @staticmethod
    def get_jina_rerank(top_n: int = 3):
        # pip install llama-index-postprocessor-jinaai-rerank
        from llama_index.postprocessor.jinaai_rerank import JinaRerank
        # 远程模型
        jina_reranker = JinaRerank(
            top_n=top_n,
            model="jina-reranker-v1-base-en",
            api_key="jina_6dd89f61b1934b1b9da71468ebbf570064hK5JQUBKI31kpNsU9JaPioBk1X"
        )
        return jina_reranker

    @staticmethod
    def get_llm_rerank(llm: Optional[LLM] = Settings.llm, top_n: int = 3):
        from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
        llm_rerank = RankGPTRerank(
            llm=llm,
            top_n=top_n
        )
        return llm_rerank
