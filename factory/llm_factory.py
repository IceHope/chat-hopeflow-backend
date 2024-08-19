from dotenv import load_dotenv

from factory.llm.hunyuan_llm_factory import HunyuanLlmFactory
from factory.llm.mini_max_lllm_factory import MinimaxLlmFactory
from factory.llm.openai.agi_llm_factory import AgiLLMFactory
from factory.llm.openai.baichuan_llm_factory import BaichuanLlmFactory
from factory.llm.openai.dashscope_llm_factory import DashscopeLlmFactory
from factory.llm.openai.deep_seek_llm_factory import DeepseekLlmFactory
from factory.llm.openai.groq_llm_factory import GroqLLMFactory
from factory.llm.openai.kimi_llm_factory import KimiLlmFactory
from factory.llm.openai.lingyi_llm_factory import LingyiLlmFactory
from factory.llm.openai.open_ai_llm_factory import OpenaiLlmFactory
from factory.llm.openai.qw2_llm_factory import QW2LLMFactory
from factory.llm.openai.zhipu_llm_factory import ZhipuLlmFactory
from factory.llm.qianfan_llm_factory import QianfanLlmFactory
from factory.llm.spark_llm_factory import SparkLLMFactory
from factory.mode_type import LLMType

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

        if mode_type == LLMType.QW2:
            return QW2LLMFactory().get_llm(mode_name)

        if mode_type == LLMType.SPARK:
            return SparkLLMFactory().get_llm(mode_name)

        if mode_type == LLMType.HUNYUAN:
            return HunyuanLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.LINGYI:
            return LingyiLlmFactory().get_llm(mode_name)

        if mode_type == LLMType.GROQ:
            return GroqLLMFactory().get_llm(mode_name)

        if mode_type in [LLMType.DOUBAO, LLMType.ANTHROPIC,
                         LLMType.MISTRAL, LLMType.GOOGLE,
                         LLMType.COHERE, LLMType.META]:
            return AgiLLMFactory().get_llm(mode_name)

    @staticmethod
    def get_default_llm():
        return LLMFactory.get_llm(LLMType.DEEPSEEK)


if __name__ == "__main__":
    llm = LLMFactory.get_llm(LLMType.DEEPSEEK)
    # llm_response = llm.invoke("介绍下你自己")
    # print(llm_response.content)
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="")
