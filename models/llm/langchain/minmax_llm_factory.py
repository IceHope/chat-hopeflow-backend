import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from models.llm.langchain.base_llm_factory import BaseLLMFactory

load_dotenv()


class MinimaxLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "abab6.5s-chat"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_community.chat_models import MiniMaxChat

        return MiniMaxChat(
            minimax_group_id=os.getenv("MINIMAX_GROUP_ID"),
            minimax_api_key=os.getenv("MINIMAX_API_KEY"),
            minimax_api_host=os.getenv("MINIMAX_API_HOST"),
            model=self.get_default_mode_name() if mode_name is None else mode_name
        )
