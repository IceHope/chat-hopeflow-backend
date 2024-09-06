import os

from dotenv import load_dotenv

from models.api_key_config import ApiKeyUrlConfig, get_groq_config
from models.llm.langchain.openai.base_openai_llm_factory import BaseOpenaiTypeFactory

load_dotenv()


def perform_groq():
    from openai import OpenAI

    # 设置 API 密钥和代理 URL
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        # base_url=os.getenv("GROQ_BASE_URL")
        base_url="https://api.openaiee.com/openai/v1",
    )

    # 发送请求
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁"},
        ],
    )

    # 打印响应
    print(response.choices[0].message.content)


class GroqLLMFactory(BaseOpenaiTypeFactory):

    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_groq_config()


if __name__ == "__main__":
    # perform_groq()
    llm = GroqLLMFactory().get_llm(mode_name="llama-3.1-8b-instant")
    result = llm.invoke("介绍下你自己")
    print(result)
