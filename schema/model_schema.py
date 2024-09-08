from typing import List

from pydantic import BaseModel, Field

from models.model_type import *
from models.names.embedding_names import *
from models.names.llm_names import *
from models.names.modal_names import *
from models.names.rerank_names import *


class ModelNameSchema(BaseModel):
    name: str = Field(default="", description="模型名称")
    input_price: float = Field(default=0.0, description="模型价格,/百万tokens")
    output_price: float = Field(default=0.0, description="模型价格,/百万tokens")


class ModelSchema(BaseModel):
    modal_type: str = Field(default="", description="模型的模态类型")
    owner_type: str = Field(default="", description="模型厂商类型")
    desc: str = Field(default="", description="具体厂商描述")
    names: list[ModelNameSchema] = Field(default=[], description="模型列表")


class ModelListSchema(BaseModel):
    llm: List[ModelSchema]
    multi_modal: List[ModelSchema]
    embedding: List[ModelSchema]
    rerank: List[ModelSchema]


def get_llm_model_names(llm_names: list) -> List[ModelNameSchema]:
    return [
        ModelNameSchema(name=item[0], input_price=item[1], output_price=item[2])
        for item in llm_names
    ]


def get_embedding_names(embedding_names: list) -> List[ModelNameSchema]:
    return [
        ModelNameSchema(name=item[0], input_price=item[1], output_price=item[1])
        for item in embedding_names
    ]


def get_rerank_names(rerank_names: list) -> List[ModelNameSchema]:
    return [
        ModelNameSchema(name=item[0], input_price=0, output_price=0)
        for item in rerank_names
    ]


def get_all_llm_models() -> List[ModelSchema]:
    return [
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.GROQ.value,
            desc="Groq",
            names=get_llm_model_names(GROQ_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.OPENAI.value,
            desc="OpenAI",
            names=get_llm_model_names(OPENAI_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.ANTHROPIC.value,
            desc="Claude",
            names=get_llm_model_names(CLAUDE_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.GOOGLE.value,
            desc="Google",
            names=get_llm_model_names(GOOGLE_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.ZHIPU.value,
            desc="智谱清言",
            names=get_llm_model_names(ZHIPU_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.DEEPSEEK.value,
            desc="DeepSeek",
            names=get_llm_model_names(DEEPSEEK_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.DASHSCOPE.value,
            desc="通义千问",
            names=get_llm_model_names(DASHSCOPE_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.QIANFAN.value,
            desc="百度千帆",
            names=get_llm_model_names(QIANFAN_NAMES),
        ),

        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.BAICHUAN.value,
            desc="百川",
            names=get_llm_model_names(BAICHUAN_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.KIMI.value,
            desc="月之暗面",
            names=get_llm_model_names(MOONSHOT_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.SPARK.value,
            desc="讯飞星火",
            names=get_llm_model_names(SPARK_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.LINGYI.value,
            desc="零一万物",
            names=get_llm_model_names(LINGYI_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.DOUBAO.value,
            desc="豆包",
            names=get_llm_model_names(DOUBAO_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.MISTRAL.value,
            desc="Mistral",
            names=get_llm_model_names(MISTRAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.LLM.value,
            owner_type=LLMType.COHERE.value,
            desc="Cohere",
            names=get_llm_model_names(COHERE_NAMES),
        ),
    ]


def get_all_multimodal_models():
    return [
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.OPENAI.value,
            desc="Openai",
            names=get_llm_model_names(OPENAI_MULTIMODAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.CLAUDE.value,
            desc="Claude",
            names=get_llm_model_names(CLAUDE_MULTIMODAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.GOOGLE.value,
            desc="Google",
            names=get_llm_model_names(GOOGLE_MULTIMODAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.DASHSCOPE.value,
            desc="通义千问",
            names=get_llm_model_names(DASHSCOPE_MULTIMODAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.ZHIPU.value,
            desc="智谱清言",
            names=get_llm_model_names(ZHIPU_MULTIMODAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.MULTIMODAL.value,
            owner_type=MutilModalType.LINGYI.value,
            desc="零一万物",
            names=get_llm_model_names(LINGYI_MULTIMODAL_NAMES),
        ),
    ]


def get_all_embedding_models():
    return [
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.OPENAI.value,
            desc="Openai",
            names=get_embedding_names(OPENAI_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.DASHSCOPE.value,
            desc="通义千问",
            names=get_embedding_names(DASHSCOPE_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.QIANFAN.value,
            desc="百度千帆",
            names=get_embedding_names(QIANFAN_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.ZHIPU.value,
            desc="智谱清言",
            names=get_embedding_names(ZHIPU_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.BAICHUAN.value,
            desc="百川",
            names=get_embedding_names(BAICHUAN_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.JINA.value,
            desc="Jina",
            names=get_embedding_names(JINA_EMBEDDING_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.EMBEDDING.value,
            owner_type=EmbeddingType.BGE.value,
            desc="BGE",
            names=get_embedding_names(BGE_EMBEDDING_NAMES),
        ),
    ]


def get_all_rerank_models():
    return [
        ModelSchema(
            modal_type=ModalType.RERANK.value,
            owner_type=RerankType.LOCAL.value,
            desc="本地模型(bge,bce)",
            names=get_rerank_names(RERANK_LOCAL_NAMES),
        ),
        ModelSchema(
            modal_type=ModalType.RERANK.value,
            owner_type=RerankType.JINA.value,
            desc="jina",
            names=get_rerank_names(RERANK_JINA_NAMES),
        ),
    ]


def get_model_list_schema() -> ModelListSchema:
    return ModelListSchema(
        llm=get_all_llm_models(),
        multi_modal=get_all_multimodal_models(),
        embedding=get_all_embedding_models(),
        rerank=get_all_rerank_models(),
    )


if __name__ == "__main__":
    print(get_all_rerank_models())
