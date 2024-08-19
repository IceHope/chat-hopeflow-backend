import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class DeepseekLlmFactory(BaseLLMFactory):

    def get_default_mode_name(self) -> str:
        return "deepseek-chat"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name
        )


if __name__ == "__main__":
    with open("tem.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    from factory.mode_names_versions import DEEPSEEK_NAMES

    llm = DeepseekLlmFactory().get_llm(DEEPSEEK_NAMES[-1][0])

    stream_response = llm.stream(prompt)
    for chunk in stream_response:
        print(chunk.content, end="")

    # response = llm.invoke(prompt)
    # print(response)
