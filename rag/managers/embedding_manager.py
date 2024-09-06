import time

from rag.config.rag_config import RagConfiguration
from rag.embedding.embedding_models import get_embedding_model, get_simple_embedding_name
from utils.log_utils import LogUtils


class EmbeddingManager:

    def __init__(
            self,
            embedding_type: str = None,
            embedding_name: str = None
    ) -> None:
        start_time = time.time()  # Start timing

        if embedding_type is None:
            embedding_config = RagConfiguration().get_embedding_config()
            LogUtils.log_info(f"Embedding config: {embedding_config}")

            self.embedding_type = embedding_config["type"]
            self.embedding_name = embedding_config["name"]
        else:
            self.embedding_type = embedding_type
            self.embedding_name = embedding_name

        self.embed_model = get_embedding_model(self.embedding_type, self.embedding_name)

        elapsed_time = round(time.time() - start_time, 2)  # Calculate elapsed time
        LogUtils.log_info(f"Embedding model loaded in {elapsed_time} seconds")

        self.embedding_dim = len(self.embed_model.get_text_embedding("hello"))

        LogUtils.log_info(f"Embedding model len ={self.embedding_dim}")

        self.simple_model_name = get_simple_embedding_name(
            embedding_type=self.embedding_type, embedding_name=self.embedding_name)

    def get_model(self):
        return self.embed_model

    def get_dim(self) -> int:
        return self.embedding_dim

    def get_model_name(self) -> str:
        return self.embedding_name

    def get_simple_model_name(self) -> str:
        return self.simple_model_name
