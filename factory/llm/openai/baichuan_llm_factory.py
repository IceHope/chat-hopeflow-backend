import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class BaichuanLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "Baichuan4"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("BAICHUAN_API_KEY"),
            base_url=os.getenv("BAICHUAN_API_BASE"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == "__main__":
    from factory.mode_names_versions import BAICHUAN_NAMES

    llm = BaichuanLlmFactory().get_llm(BAICHUAN_NAMES[-1][0])
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="|")
