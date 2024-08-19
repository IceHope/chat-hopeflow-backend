from enum import Enum
from pprint import pprint


class LLMType(str, Enum):
    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    QIANFAN = "qianfan"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    BAICHUAN = "baichuan"
    MINIMAX = "minimax"
    KIMI = "kimi"
    SPARK = "spark"
    LINGYI = "lingyi"
    HUNYUAN = "hunyuan"
    QW2 = "qw2"
    GROQ = "groq"

    DOUBAO = "doubao"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    GOOGLE = "google"
    COHERE = "cohere"
    META = "meta"

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


class EmbeddingType(str, Enum):
    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    QIANFAN = "qianfan"
    ZHIPU = "zhipu"
    BAICHUAN = "baichuan"


if __name__ == "__main__":
    dict = LLMType.dict()
    pprint(dict)
    # print(LLMType.get_enum("openai"))
