from dotenv import load_dotenv

from models.llm.langchain.minmax_llm_factory import MinimaxLlmFactory
from models.llm.langchain.openai.baichuan_llm_factory import BaichuanLlmFactory
from models.llm.langchain.openai.base_agi_llm_factory import BaseAgiLLMFactory
from models.llm.langchain.openai.dashscope_llm_factory import DashscopeLlmFactory
from models.llm.langchain.openai.deep_seek_llm_factory import DeepseekLlmFactory
from models.llm.langchain.openai.groq_llm_factory import GroqLLMFactory
from models.llm.langchain.openai.kimi_llm_factory import KimiLlmFactory
from models.llm.langchain.openai.lingyi_llm_factory import LingyiLlmFactory
from models.llm.langchain.openai.open_ai_llm_factory import OpenaiLlmFactory
from models.llm.langchain.openai.silicon_llm_factory import SiliconLLMFactory
from models.llm.langchain.openai.zhipu_llm_factory import ZhipuLlmFactory
from models.llm.langchain.qianfan_llm_factory import QianfanLlmFactory
from models.llm.langchain.spark_llm_factory import SparkLLMFactory
from models.model_type import LLMType

load_dotenv()


class LLMFactory:
    @staticmethod
    def get_llm(mode_type: LLMType, mode_name: str = None):
        if mode_type == LLMType.OPENAI:
            return OpenaiLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.QIANFAN:
            return QianfanLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.DASHSCOPE:
            return DashscopeLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.ZHIPU:
            return ZhipuLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.MINIMAX:
            return MinimaxLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.KIMI:
            return KimiLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.DEEPSEEK:
            return DeepseekLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.BAICHUAN:
            return BaichuanLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.SILICONFLOW:
            return SiliconLLMFactory().get_llm(mode_name)

        if mode_type == LLMType.SPARK:
            return SparkLLMFactory().get_llm(mode_name)

        if mode_type == LLMType.LINGYI:
            return LingyiLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.GROQ:
            return GroqLLMFactory().get_llm(mode_name)

        if mode_type in [LLMType.DOUBAO, LLMType.ANTHROPIC,
                         LLMType.MISTRAL, LLMType.GOOGLE,
                         LLMType.COHERE, ]:
            return BaseAgiLLMFactory().get_llm(mode_name)

    @staticmethod
    def get_default_llm():
        return LLMFactory.get_llm(LLMType.DEEPSEEK)


if __name__ == "__main__":
    llm = LLMFactory.get_llm(LLMType.OPENAI)
    # llm_response = llm.invoke("介绍下你自己")
    # print(llm_response.content)
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="")
