import os

from dotenv import load_dotenv
from pydantic import BaseModel

from models.model_type import LLMType

load_dotenv()


def get_jina_api_key():
    return os.getenv("JINA_API_KEY")


def get_langsmith_api_key():
    return os.getenv("LANGSMITH_API_KEY")


class ApiKeyUrlConfig(BaseModel):
    api_key: str
    base_url: str
    default_model_name: str


def get_openai_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        default_model_name="gpt-4o",
    )


def get_agi_type_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("AGI_API_KEY"),
        base_url=os.getenv("AGI_BASE_URL"),
        default_model_name="gemini-1.5-flash"
    )


def get_groq_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_BASE_URL"),
        default_model_name="llama-3.1-70b-versatile"
    )


def get_zhipu_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        default_model_name="GLM-4-Flash"
    )


def get_dashscope_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        default_model_name="qwen-plus"
    )


def get_baichuan_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("BAICHUAN_API_KEY"),
        base_url=os.getenv("BAICHUAN_BASE_URL"),
        default_model_name="Baichuan3-Turbo"
    )


def get_kimi_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL"),
        default_model_name="moonshot-v1-8k"
    )


def get_deepseek_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        default_model_name="deepseek-chat"
    )


def get_lingyi_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("LINGYI_API_KEY"),
        base_url=os.getenv("LINGYI_BASE_URL"),
        default_model_name="yi-medium"
    )


def get_siliconflow_config():
    return ApiKeyUrlConfig(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        default_model_name="meta-llama/Meta-Llama-3-70B-Instruct"
    )
