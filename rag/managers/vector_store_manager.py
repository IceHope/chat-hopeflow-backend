# 导入所需的库和模块
from time import time
from llama_index.vector_stores.milvus.base import MilvusVectorStore
from rag.db.milvus.vector_store import load_hybrid_milvus, load_single_milvus
from utils.log_utils import LogUtils
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

from llama_index.core.schema import BaseNode


def _get_milvus_vector_store(
    collection_name: str, embedding_size: int
) -> MilvusVectorStore:
    # 加载并返回Milvus向量存储
    LogUtils.log_info(
        f"正在加载(Milvus)向量数据库，集合名称：{collection_name}，嵌入维度：{embedding_size}"
    )
    # vector_store = load_hybrid_milvus(collection_name, embedding_size)
    vector_store = load_single_milvus(
        overwrite=False, collection_name=collection_name, dim=embedding_size
    )
    return vector_store


class VectorStoreManager:

    def __init__(
        self,
        collection_name: str,
        embedding_size: int,
    ) -> None:
        LogUtils.log_info(
            f"初始化 VectorStoreManager，集合名称：{collection_name}，嵌入维度：{embedding_size}"
        )
        # 初始化向量存储
        self.vector_store = _get_milvus_vector_store(collection_name, embedding_size)
        LogUtils.log_info("VectorStoreManager初始化完成")

    def load_nodes(self, nodes: list[BaseNode]) -> None:
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 构建向量索引
        start_time = time()  # 开始计时
        LogUtils.log_info("开始构建向量索引")
        VectorStoreIndex(
            nodes=nodes, storage_context=storage_context, show_progress=True
        )
        elapsed_time = round(time() - start_time, 2)  # 计算耗时
        LogUtils.log_info(f"向量索引构建完成，耗时：{elapsed_time}秒")

    def get_vectore_store(self):
        return self.vector_store
