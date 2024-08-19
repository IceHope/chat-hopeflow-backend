from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


class BaseLLMFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_default_mode_name(self) -> str:
        pass

    @abstractmethod
    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        pass
