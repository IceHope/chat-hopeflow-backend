from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, MetadataFilters

from rag.retriver.hope_retriever import HopeRetriever
from utils.log_utils import LogUtils


class RetrieverManager:
    def __init__(
            self, vector_store: BasePydanticVectorStore
    ) -> None:
        self.vector_store = vector_store

        self.hope_retriever = HopeRetriever(
            vector_store=vector_store,
        )

    def retrieve_chunk(self, query: str, filters: MetadataFilters = None) -> List[NodeWithScore]:
        nodes = self.hope_retriever.retrieve(query=query, filters=filters)
        LogUtils.log_info(f"retrieve_chunk {len(nodes)} nodes: ")
        for node in nodes:
            LogUtils.log_info(f"{node.node_id} : {node.score}")
        return nodes
