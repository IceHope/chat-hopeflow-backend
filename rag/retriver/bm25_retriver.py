from typing import List

from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever


# llamaindix定义的bm25必须得传入所有nodes才行,效率肯定不如数据库本身支持bm25的效率高
# 比如weaviate,所以这里只是简单包装一下
def get_bm25_retriever(nodes: List[BaseNode], similarity_top_k=3) -> BM25Retriever:
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)
    return bm25_retriever
