import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory

load_dotenv()


def _perform_openai():
    from openai import OpenAI

    # 设置 API 密钥和代理 URL
    client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        # base_url=os.getenv("GOOGLE_BASE_URL")
        base_url="https://api.openaiee.com/v1beta"
    )

    # 发送请求
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁"}
        ]
    )

    # 打印响应
    print(response.choices[0].message.content)


class GeminiLLMFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        pass

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url=os.getenv("GOOGLE_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == "__main__":
    # llm = GeminiLLMFactory().get_llm(mode_name="gemini-1.5-flash")
    # result = llm.invoke("你好")
    # print(result)
    _perform_openai()
