from utils.log_utils import LogUtils

BGE_EMBEDDINGS = [
    ("bge-small-zh-v1.5", "F:/HuggingFace/Embedding/bge-small-zh-v1.5", 512, "bge_small_zh"),
    ("bge-small-en-v1.5", "F:/HuggingFace/Embedding/bge-small-en-v1.5", 384, "bge_small_en"),
    ("bge-large-zh-v1.5", "F:/HuggingFace/Embedding/bge-large-zh-v1.5", 1024, "bge_large_zh"),
    ("bge-large-en-v1.5", "F:/HuggingFace/Embedding/bge-large-en-v1.5", 1024, "bge_large_en"),
    ("bge-m3", "F:/HuggingFace/Embedding/bge-m3", 1024, "bge_m3"),
]


# 返回元组类型
def get_bge_config(mode_name: str):
    for item in BGE_EMBEDDINGS:
        if item[0] == mode_name:
            return item
    raise ValueError(f"Unknown BGE embedding: {mode_name}")


def get_bge_embedding_model(bge_mode_name: str):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embedding_item = get_bge_config(bge_mode_name)

    embed_model = HuggingFaceEmbedding(model_name=embedding_item[1])

    LogUtils.log_info(f"bge_embedding model:  {bge_mode_name} ")
    return embed_model
