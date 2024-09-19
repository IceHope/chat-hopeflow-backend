from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from models.api_key_config import ApiKeyUrlConfig
from models.llm.langchain.base_llm_factory import BaseLLMFactory


class BaseOpenaiTypeFactory(BaseLLMFactory, ABC):
    @abstractmethod
    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        pass

    def get_default_mode_name(self) -> str:
        return self.get_api_key_url_config().default_model_name

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        # !pip install --upgrade langchain-openai
        from langchain_openai import ChatOpenAI

        _config = self.get_api_key_url_config()

        return ChatOpenAI(
            api_key=_config.api_key,
            base_url=_config.base_url,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )
