import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class KimiLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "moonshot-v1-8k"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url=os.getenv("KIMI_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == "__main__":
    from factory.mode_names_versions import MOONSHOT_NAMES

    llm = KimiLlmFactory().get_llm(MOONSHOT_NAMES[0][0])
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="|")
