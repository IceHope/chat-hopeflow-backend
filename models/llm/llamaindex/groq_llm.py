from llama_index.core.llms import LLM

from models.api_key_config import ApiKeyUrlConfig, get_groq_config
from models.llm.langchain.base_llm_factory import BaseLLMLlamaFactory
from llama_index.llms.groq import Groq


class GroqLlmaFactory(BaseLLMLlamaFactory):
    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_groq_config()

    def get_llm(self, mode_name: str = None) -> LLM:
        _config = self.get_api_key_url_config()

        return Groq(
            model=_config.default_model_name if mode_name is None else mode_name,
            api_key=_config.api_key,
            api_base=_config.base_url,
            temperature=0.7,
        )
