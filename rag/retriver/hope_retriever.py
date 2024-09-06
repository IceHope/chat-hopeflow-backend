from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
)
from schema.rag_config import RagFrontendConfig


class HopeRetriever:
    def __init__(self, vector_store: BasePydanticVectorStore) -> None:
        vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        # similarity_top_k=6 是每个query最原始的返回个数
        self.base_retriever = vector_store_index.as_retriever(
            similarity_top_k=6
        )

        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[
                self.base_retriever,
            ],
            mode=FUSION_MODES.RECIPROCAL_RANK,  # RRF
            num_queries=3,  # 生成 query数 ,包含了原始查询
            use_async=False,  # 协程查询
            similarity_top_k=6,  # RRF排序后保留的个数,跟原始的个数保持一致
            # query_gen_prompt="...",  # 可以自定义 query 生成的 prompt 模板,比如中英文混合的query
        )

    def retrieve(
            self,
            query: str,
            rag_config: RagFrontendConfig = None,
            filters: MetadataFilters = None,
    ) -> List[NodeWithScore]:
        # filters = MetadataFilters(
        #     filters=[ExactMatchFilter(key="file_name", value="uber_2021.pdf")]
        # )
        if rag_config:
            self.base_retriever._similarity_top_k = rag_config.rag_retrieve_count
            self.fusion_retriever.similarity_top_k = rag_config.rag_retrieve_count
            self.fusion_retriever.num_queries = rag_config.rag_fusion_count

        if filters:
            self.base_retriever._filters = filters
            self.fusion_retriever._retrievers = [self.base_retriever]

        return self.fusion_retriever.retrieve(QueryBundle(query_str=query))

    # async def _multi_retrieve(
    #         self, queries: List[str]
    # ) -> Dict[str, List[NodeWithScore]]:
    #     LogUtils.log_info(f"multi_retrieve: {queries}")
    #     tasks = []
    #     for query in queries:
    #         tasks.append(self.query_engine.aretrieve(QueryBundle(query_str=query)))
    #
    #     task_results = await asyncio.gather(*tasks)
    #
    #     results = {}
    #     for query, query_result in zip(queries, task_results):
    #         LogUtils.log_info(f"query: {query}, result: {query_result}")
    #         results[query] = query_result
    #
    #     return results
    #
    # def multi_retrieve(self, queries: List[str]):
    #     return asyncio.run(self._multi_retrieve(queries))
