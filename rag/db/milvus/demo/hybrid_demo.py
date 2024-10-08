import time

from llama_index.core import Settings

from rag.db.milvus.vector_store import load_hybrid_milvus
from rag.managers.embedding_manager import EmbeddingManager
from utils.log_utils import LogUtils

embedding_manager = EmbeddingManager("bge-small-zh-v1.5")


def _get_nodes():
    from llama_index.core import SimpleDirectoryReader

    # Load data
    documents = SimpleDirectoryReader(
        input_files=["./data/paul_graham_essay.txt"]
    ).load_data()

    print("documents len=", len(documents))

    from llama_index.core.node_parser import SentenceWindowNodeParser

    # Create the sentence window node parser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Extract nodes from documents
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    print("nodes len =", len(nodes))
    return nodes


def load_data():
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex

    _init_embedding()
    nodes = _get_nodes()
    vector_store = load_hybrid_milvus(
        collection_name="hybrid_hope_demo",
        overwrite=True,
        dim=384
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)


if __name__ == "__main__":
    load_data()
