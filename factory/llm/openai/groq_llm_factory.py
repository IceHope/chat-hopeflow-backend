import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


def perform_groq():
    from openai import OpenAI

    # 设置 API 密钥和代理 URL
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_BASE_URL")
    )

    # 发送请求
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )

    # 打印响应
    print(response.choices[0].message.content)


class GroqLLMFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        pass

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == "__main__":
    perform_groq()
    # llm = GroqLLMFactory().get_llm(mode_name="llama-3.1-8b-instant")
    # result = llm.invoke("介绍下你自己")
    # print(result)
