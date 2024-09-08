from models.api_key_config import ApiKeyUrlConfig, get_baichuan_config
from models.llm.langchain.openai.base_openai_llm_factory import BaseOpenaiTypeFactory


class BaichuanLlmFactory(BaseOpenaiTypeFactory):

    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_baichuan_config()