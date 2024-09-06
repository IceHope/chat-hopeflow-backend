RATE = 7  # 汇率
OPENAI_EMBEDDING_NAMES = [
    ("text-embedding-3-large", 0.13 * RATE, "openai_3_large"),
    ("text-embedding-3-small", 0.02 * RATE, "openai_3_small"),
]

DASHSCOPE_EMBEDDING_NAMES = [
    ("text-embedding-v2", 0.7, "dashscope_v2"),
]
QIANFAN_EMBEDDING_NAMES = [
    ("Embedding-V1", 2, "qianfan_v1"),
]
ZHIPU_EMBEDDING_NAMES = [
    ("embedding-2", 0.5, "zhipu_2"),
    ("embedding-3", 0.5, "zhipu_3"),
]

BAICHUAN_EMBEDDING_NAMES = [
    ("Baichuan-Text-Embedding", 0.5, "baichuan"),
]
JINA_EMBEDDING_NAMES = [
    ("jina-embeddings-v2-base-en", 0.5, "jina_2_en"),
    ("jina-embeddings-v2-base-zh", 0.5, "jina_2_zh"),
    ("jina-clip-v1", 0.5, "jina_clip"),
]
BGE_EMBEDDING_NAMES = [
    ("bge-m3", 0, "bge_m3", 1024, "EMBEDDING_BGE_M3_PATH"),
    ("bge-large-zh-v1.5", 0, "bge_large_zh", 1024, "EMBEDDING_BGE_LARGE_ZH_PATH"),
    ("bge-large-en-v1.5", 0, "bge_large_en", 1024, "EMBEDDING_BGE_LARGE_EN_PATH"),
    ("bge-small-zh-v1.5", 0, "bge_small_zh", 512, "EMBEDDING_BGE_SMALL_ZH_PATH"),
    ("bge-small-en-v1.5", 0, "bge_small_en", 384, "EMBEDDING_BGE_SMALL_EN_PATH"),
]
