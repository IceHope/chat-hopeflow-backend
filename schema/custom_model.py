from typing import List

from pydantic import BaseModel, Field

from factory.mode_names_versions import *
from factory.mode_type import LLMType


class CustomModelName(BaseModel):
    name: str = Field(default="", description="模型名称")
    input_price: float = Field(default=0.0, description="模型价格,/百万tokens")
    output_price: float = Field(default=0.0, description="模型价格,/百万tokens")


class CustomModel(BaseModel):
    type: str = Field(default="", description="模型类型")
    desc: str = Field(default="", description="具体厂商描述")
    names: list[CustomModelName] = Field(default=[], description="模型列表")


def get_custom_model_names(names_versions: list) -> List[CustomModelName]:
    return [CustomModelName(name=item[0], input_price=item[1], output_price=item[2])
            for item in names_versions]


def get_all_custom_models() -> List[CustomModel]:
    return [
        CustomModel(type=LLMType.GROQ.value, desc="Groq", names=get_custom_model_names(GROQ_NAMES)),
        CustomModel(type=LLMType.DEEPSEEK.value, desc="DeepSeek", names=get_custom_model_names(DEEPSEEK_NAMES)),
        CustomModel(type=LLMType.OPENAI.value, desc="OpenAI", names=get_custom_model_names(OPENAI_NAMES)),
        CustomModel(type=LLMType.DASHSCOPE.value, desc="通义千问", names=get_custom_model_names(DASHSCOPE_NAMES)),
        CustomModel(type=LLMType.QIANFAN.value, desc="百度千帆", names=get_custom_model_names(QIANFAN_NAMES)),
        CustomModel(type=LLMType.ZHIPU.value, desc="智谱清言", names=get_custom_model_names(ZHIPU_NAMES)),
        CustomModel(type=LLMType.BAICHUAN.value, desc="百川", names=get_custom_model_names(BAICHUAN_NAMES)),
        CustomModel(type=LLMType.KIMI.value, desc="月之暗面", names=get_custom_model_names(MOONSHOT_NAMES)),
        CustomModel(type=LLMType.SPARK.value, desc="讯飞星火", names=get_custom_model_names(SPARK_NAMES)),
        CustomModel(type=LLMType.LINGYI.value, desc="零一万物", names=get_custom_model_names(LINGYI_NAMES)),
        CustomModel(type=LLMType.HUNYUAN.value, desc="腾讯混元", names=get_custom_model_names(HUANYUAN_NAMES)),
        CustomModel(type=LLMType.DOUBAO.value, desc="豆包", names=get_custom_model_names(DOUBAO_NAMES)),
        CustomModel(type=LLMType.ANTHROPIC.value, desc="Claude", names=get_custom_model_names(CLAUDE_NAMES)),
        CustomModel(type=LLMType.MISTRAL.value, desc="Mistral", names=get_custom_model_names(MISTRAL_NAMES)),
        CustomModel(type=LLMType.GOOGLE.value, desc="Google", names=get_custom_model_names(GOOGLE_NAMES)),
        CustomModel(type=LLMType.COHERE.value, desc="Cohere", names=get_custom_model_names(COHERE_NAMES)),
        CustomModel(type=LLMType.META.value, desc="Llama", names=get_custom_model_names(META_NAMES)),
    ]


if __name__ == '__main__':
    print(get_all_custom_models())
