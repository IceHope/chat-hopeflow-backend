import time
from typing import Optional, List

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.milvus import MilvusVectorStore

from rag.rag_init import RagInit
from rag.sparse_bedding import ExampleSparseEmbeddingFunction
from utils.log_utils import LogUtils


def get_document(input_dir: Optional[str] = None, input_files: Optional[List] = None):
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        input_files=input_files,
        recursive=True,
        # # required_exts=[".pdf", ".txt", ".md"],
        # required_exts=[".md"],
        file_extractor={".pdf": PyMuPDFReader(), ".json": JSONReader()},
    ).load_data(
        show_progress=True,
    )

    print("len(documents)= ", len(documents))

    return documents


def load_python_data():
    documents = SimpleDirectoryReader(
        input_dir="F:/AiData/agi-ta/python-guide",
        file_extractor={".json": JSONReader()},
    ).load_data(
        show_progress=True,
    )
    print("len(documents) = ", len(documents))

    node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)
    print("len(nodes) = ", len(nodes))
    return nodes


def load_milvus():
    start_time = time.time()
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="python_demo",
        dim=1024,
        overwrite=True,
        sparse_embedding_function=ExampleSparseEmbeddingFunction(),
        enable_sparse=True,
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    LogUtils.log_info("load_milvus time: ", time.time() - start_time)
    return vector_store


def main():
    nodes = load_python_data()

    vector_store = load_milvus()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    start_time = time.time()
    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, show_progress=True
    )
    LogUtils.log_info("index time: ", time.time() - start_time)


if __name__ == "__main__":
    RagInit.init()

    main()
