from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from llama_index.core.llms import LLM

from models.api_key_config import ApiKeyUrlConfig


class BaseLLMFactory(ABC):
    def get_default_mode_name(self) -> str:
        pass

    @abstractmethod
    def get_llm(self, mode_name: str = None) -> Any:
        pass


class BaseLLMLcFactory(BaseLLMFactory):
    @abstractmethod
    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        pass


class BaseLLMLlamaFactory(BaseLLMFactory):
    @abstractmethod
    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        pass

    @abstractmethod
    def get_llm(self, mode_name: str = None) -> LLM:
        pass
