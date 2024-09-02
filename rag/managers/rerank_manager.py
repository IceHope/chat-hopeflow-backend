import time
from typing import List

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from rag.rerank.reranker import RagReranker
from utils.log_utils import LogUtils
from controller.rag.rag_config import RagFrontendConfig


class RerankManager:
    def __init__(self):
        start_time = time.time()
        self.rerank_model = RagReranker.get_transformer_cross_encoder(3)

        elapsed_time = round(time.time() - start_time, 2)
        LogUtils.log_info(f"RerankManager init elapsed_time: {elapsed_time} seconds")

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        rag_config: RagFrontendConfig = None,
    ) -> List[NodeWithScore]:
        if rag_config:
            self.rerank_model.top_n = rag_config.rag_rerank_count

        rerank_nodes = self.rerank_model.postprocess_nodes(query_str=query, nodes=nodes)

        LogUtils.log_info(f"rerank {len(rerank_nodes)} nodes: ")
        for node in rerank_nodes:
            LogUtils.log_info(f"{node.node_id} : {node.score}")
        return rerank_nodes
