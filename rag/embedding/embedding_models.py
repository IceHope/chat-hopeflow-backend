import os

from dotenv import load_dotenv

from models.model_type import EmbeddingType
from models.names.embedding_names import (
    BGE_EMBEDDING_NAMES,
    JINA_EMBEDDING_NAMES,
    ZHIPU_EMBEDDING_NAMES,
)

load_dotenv()


def get_simple_embedding_name(embedding_type: str, embedding_name: str):
    names_list = []
    if embedding_type == EmbeddingType.BGE:
        names_list = BGE_EMBEDDING_NAMES
    if embedding_type == EmbeddingType.ZHIPU:
        names_list = ZHIPU_EMBEDDING_NAMES
    if embedding_type == EmbeddingType.JINA:
        names_list = JINA_EMBEDDING_NAMES

    if len(names_list) == 0:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    for item in names_list:
        if item[0] == embedding_name:
            return item[2]

    raise ValueError(f"Unknown embedding_name name: {embedding_name}")


def get_zhipu_embedding_model(mode_name: str = None):
    from rag.embedding.zhipu_embedding import ZhipuEmbedding

    return ZhipuEmbedding(model_name=mode_name, api_key=os.getenv("ZHIPUAI_API_KEY"))


def get_bge_embedding_local_path(embedding_name: str):
    for item in BGE_EMBEDDING_NAMES:
        if item[0] == embedding_name:
            return os.getenv(item[-1])

    raise ValueError(f"Unknown embedding name: {embedding_name}")


def get_bge_embedding_model(bge_mode_name: str):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    local_path = get_bge_embedding_local_path(bge_mode_name)

    embed_model = HuggingFaceEmbedding(model_name=local_path)

    return embed_model


def get_jina_embedding_model(jina_mode_name: str = None):
    from llama_index.embeddings.jinaai import JinaEmbedding

    return JinaEmbedding(
        api_key=os.getenv("JINA_API_KEY"),
        model=jina_mode_name,
    )


def get_embedding_model(embedding_type: str, embedding_name: str):
    if embedding_type == EmbeddingType.BGE:
        return get_bge_embedding_model(bge_mode_name=embedding_name)

    if embedding_type == EmbeddingType.ZHIPU:
        return get_zhipu_embedding_model(mode_name=embedding_name)

    if embedding_type == EmbeddingType.JINA:
        return get_jina_embedding_model(jina_mode_name=embedding_name)
