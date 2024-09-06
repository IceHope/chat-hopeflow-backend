from llama_index.core.llms import LLM

from models.api_key_config import *
from models.llm.langchain.base_llm_factory import BaseLLMLlamaFactory


class OpenaiLlamaFactory(BaseLLMLlamaFactory):
    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_openai_config()

    def get_llm(self, mode_name: str = None) -> LLM:
        _config = self.get_api_key_url_config()

        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=_config.default_model_name if mode_name is None else mode_name,
            api_key=_config.api_key,
            api_base=_config.base_url,
            temperature=0.7,
        )
