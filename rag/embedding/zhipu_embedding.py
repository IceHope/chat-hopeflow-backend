from enum import Enum
from typing import Optional, Any, List, Union

from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding

load_dotenv()


class ZhipuTextEmbeddingModels(str, Enum):
    """Zhipu TextEmbedding models."""
    TEXT_EMBEDDING_V2 = "embedding-2"
    TEXT_EMBEDDING_V3 = "embedding-3"


EMBED_MAX_BATCH_SIZE = 10


def get_text_embedding(
        model: str,
        text: Union[str, List[str]],
        api_key: Optional[str] = None,
) -> List[List[float]]:
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("ZhipuAI requires `pip install zhipuai")

    if isinstance(text, str):
        text = [text]

    client = ZhipuAI(api_key=api_key)
    resp = client.embeddings.create(
        model=model,
        input=text,
    )

    return [r.embedding for r in resp.data]


class ZhipuEmbedding(BaseEmbedding):

    def __init__(
            self,
            model_name: str = ZhipuTextEmbeddingModels.TEXT_EMBEDDING_V2,
            text_type: str = "document",
            api_key: Optional[str] = None,
            embed_batch_size: int = EMBED_MAX_BATCH_SIZE,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._api_key = api_key
        self._text_type = text_type

    @classmethod
    def class_name(cls) -> str:
        return "ZhipuEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        emb = get_text_embedding(
            self.model_name,
            query,
            api_key=self._api_key,
        )
        if len(emb) > 0 and emb[0] is not None:
            return emb[0]
        else:
            return []

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        emb = get_text_embedding(
            self.model_name,
            text,
            api_key=self._api_key,
        )
        if len(emb) > 0 and emb[0] is not None:
            return emb[0]
        else:
            return []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return get_text_embedding(
            self.model_name,
            texts,
            api_key=self._api_key,
        )

    # TODO: use proper async methods
    async def _aget_text_embedding(self, query: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embedding(query)

    # TODO: user proper async methods
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_query_embedding(query)
