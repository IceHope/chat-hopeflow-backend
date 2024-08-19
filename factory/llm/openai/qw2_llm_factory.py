import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class QW2LLMFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "alibaba/Qwen2-72B-Instruct"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL"),
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == "__main__":
    llm = QW2LLMFactory().get_llm()
    response = llm.invoke("你是谁")
    print(response)
