from enum import Enum
from pprint import pprint


class ModalType(str, Enum):
    LLM = "llm"
    MULTIMODAL = "multi_modal"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class LLMType(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    DASHSCOPE = "dashscope"
    QIANFAN = "qianfan"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    BAICHUAN = "baichuan"
    MINIMAX = "minimax"
    KIMI = "kimi"
    SPARK = "spark"
    LINGYI = "01"
    SILICONFLOW = "siliconflow"

    DOUBAO = "doubao"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    GOOGLE = "google"
    COHERE = "cohere"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def dict(cls):
        return [(item, item.value) for item in cls]

    @classmethod
    def get_enum_from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item


class MutilModalType(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GOOGLE = "google"
    DASHSCOPE = "dashscope"
    ZHIPU = "zhipu"
    LINGYI = "lingyi"


class EmbeddingType(str, Enum):
    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    QIANFAN = "qianfan"
    ZHIPU = "zhipu"
    BAICHUAN = "baichuan"
    BGE = "bge"
    JINA = "jina"


class RerankType(str, Enum):
    LOCAL = "local"
    JINA = "jina"


if __name__ == "__main__":
    dict = LLMType.dict()
    pprint(dict)
    # print(LLMType.get_enum("openai"))
