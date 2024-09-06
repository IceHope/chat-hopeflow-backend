from time import time

from llama_index.core import VectorStoreIndex, StorageContext

from rag.config.rag_config import RagConfiguration
from rag.db.milvus.demo.custom_vector_store import CustomMilvusVectorStore
from rag.managers.chunk_manager import ChunkManager
from rag.managers.embedding_manager import EmbeddingManager
from rag.managers.reader_manager import ReaderManager
from utils.log_utils import LogUtils

reader_manager = ReaderManager()

chunk_manager = ChunkManager()

embedding_manager = EmbeddingManager("bge-small-en-v1.5")

collection_name = "partition_test"


def load_partition():
    paths = ["./data/paul_graham_essay.txt"]
    documents = reader_manager.load_file_list(input_file_paths=paths)
    nodes = chunk_manager.chunk_documents(documents=documents)

    vector_store = CustomMilvusVectorStore(
        uri=RagConfiguration().get_milvus_uri(),
        collection_name=collection_name,
        # dim=embedding_manager.get_dim(),
        dim=384,
        overwrite=False,
    )

    client = vector_store.client
    res = client.list_partitions(collection_name=collection_name)
    print(res)

    client.create_partition(
        collection_name=collection_name,
        partition_name="partitionC"
    )

    res = client.list_partitions(collection_name=collection_name)
    print(res)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    start_time = time()  # 开始计时
    LogUtils.log_info("开始构建向量索引")

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    elapsed_time = round(time() - start_time, 2)  # 计算耗时
    LogUtils.log_info(f"向量索引构建完成，耗时：{elapsed_time}秒")


def query_partition():
    vector_store = CustomMilvusVectorStore(
        uri=RagConfiguration().get_milvus_uri(),
        collection_name=collection_name,
        # dim=embedding_manager.get_dim(),
        dim=384,
        overwrite=False,
    )
    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    # similarity_top_k=6 是每个query最原始的返回个数
    base_retriever = vector_store_index.as_retriever(
        similarity_top_k=3,
    )
    nodes = base_retriever.retrieve("What I Worked On February 2021")
    print(len(nodes))
    # for node in nodes:
    #     print("-----------")
    #     print(node.text)


if __name__ == "__main__":
    # load_partition()
    query_partition()
