import time

from llama_index.vector_stores.milvus import MilvusVectorStore

from rag.config.rag_config import RagConfiguration
from rag.db.milvus.sparse_bedding import LocalSparseEmbeddingFunction
from utils.log_utils import LogUtils


def load_single_milvus(
        overwrite=False, collection_name: str = "hope_test", dim: int = 1024
) -> MilvusVectorStore:
    start_time = time.time()
    vector_store = MilvusVectorStore(
        uri=RagConfiguration().get_milvus_uri(),
        collection_name=collection_name,
        dim=dim,
        overwrite=overwrite,
    )
    elapsed_time = round(time.time() - start_time, 2)
    LogUtils.log_info(f"load_single_milvus time: {elapsed_time} seconds")
    return vector_store


def load_hybrid_milvus(
        overwrite=False, collection_name: str = "hope_test", dim: int = 1024
) -> MilvusVectorStore:
    start_time = time.time()
    vector_store = MilvusVectorStore(
        uri=RagConfiguration().get_milvus_uri(),
        collection_name=collection_name,
        dim=dim,
        overwrite=overwrite,
        sparse_embedding_function=LocalSparseEmbeddingFunction(),
        enable_sparse=True,
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    elapsed_time = round(time.time() - start_time, 2)
    LogUtils.log_info(f"load_hybrid_milvus time: {elapsed_time} seconds")
    return vector_store
