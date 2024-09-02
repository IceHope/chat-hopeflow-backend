import time

from llama_index.core import Settings

from rag.config.rag_config import RagConfiguration
from rag.embedding.bge_embedding import get_bge_embedding_model, get_bge_config
from utils.log_utils import LogUtils


class EmbeddingManager:

    def __init__(self) -> None:
        start_time = time.time()  # Start timing

        bge_embedding_name = RagConfiguration().get_bge_embedding_config()

        self.model_name = bge_embedding_name
        self.embed_model = get_bge_embedding_model(bge_embedding_name)
        Settings.embed_model = self.embed_model

        elapsed_time = round(time.time() - start_time, 2)  # Calculate elapsed time
        LogUtils.log_info(f"Embedding model loaded in {elapsed_time} seconds")
        LogUtils.log_info(
            "Embedding model len = ",
            len(Settings.embed_model.get_text_embedding("hello")),
        )

        bge_embedding_config = get_bge_config(bge_embedding_name)

        self.embedding_dim = bge_embedding_config[2]
        self.simple_model_name = bge_embedding_config[3]

    def get_model(self):
        return self.embed_model

    def get_dim(self) -> int:
        return self.embedding_dim

    def get_model_name(self) -> str:
        return self.model_name

    def get_simple_model_name(self) -> str:
        return self.simple_model_name
